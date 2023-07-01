"""
Vision Transformer (ViT) in PyTorch.
"""

import math
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.helpers import named_apply, adapt_input_conv
from timm.models.layers import Mlp, DropPath, trunc_normal_, lecun_normal_

from lib.models.layers.patch_embed import PatchEmbed
from lib.models.grm.base_backbone import BaseBackbone


class Attention(nn.Module):
    def __init__(self, dim, num_heads, qkv_bias, attn_drop=0., proj_drop=0., divide=False, gauss=False, early=True):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.divide = divide
        self.gauss = gauss
        self.early = early
        if self.divide:
            if not self.early:
                self.divide_global_transform = nn.Sequential(
                    nn.Linear(dim, 384)
                )
                self.divide_local_transform = nn.Sequential(
                    nn.Linear(dim, 384)
                )
            self.divide_predict = nn.Sequential(
                nn.Linear(dim * 2, 384) if self.early else nn.Identity(),
                nn.GELU(),
                nn.Linear(384, 192),
                nn.GELU(),
                nn.Linear(192, 2),
                nn.Identity() if self.gauss else nn.LogSoftmax(dim=-1)
            )
            if self.gauss:
                self.divide_gaussian_filter = nn.Conv2d(2, 2, kernel_size=5, stride=1, padding=2)
                self.init_gaussian_filter()
                self.divide_gaussian_filter.requires_grad = False

    def init_gaussian_filter(self, k_size=5, sigma=1.0):
        center = k_size // 2
        x, y = np.mgrid[0 - center: k_size - center, 0 - center: k_size - center]
        gaussian_kernel = 1 / (2 * np.pi * sigma) * np.exp(-(np.square(x) + np.square(y)) / (2 * np.square(sigma)))
        self.divide_gaussian_filter.weight.data = torch.Tensor(gaussian_kernel).to(
            self.divide_gaussian_filter.weight.device).unsqueeze(0).unsqueeze(0).repeat(2, 2, 1, 1)
        self.divide_gaussian_filter.bias.data.zero_()

    def softmax_with_policy(self, attn, policy, eps=1e-6):
        B, H, N, N = attn.size()
        group1 = policy[:, :, 0].reshape(B, 1, N, 1) @ policy[:, :, 0].reshape(B, 1, 1, N)
        group2 = policy[:, :, 1].reshape(B, 1, N, 1) @ policy[:, :, 1].reshape(B, 1, 1, N)
        group3 = policy[:, :, 0].reshape(B, 1, N, 1) @ policy[:, :, 2].reshape(B, 1, 1, N) + \
                 policy[:, :, 1].reshape(B, 1, N, 1) @ policy[:, :, 2].reshape(B, 1, 1, N) + \
                 policy[:, :, 2].reshape(B, 1, N, 1) @ policy[:, :, 2].reshape(B, 1, 1, N) + \
                 policy[:, :, 2].reshape(B, 1, N, 1) @ policy[:, :, 0].reshape(B, 1, 1, N) + \
                 policy[:, :, 2].reshape(B, 1, N, 1) @ policy[:, :, 1].reshape(B, 1, 1, N)
        attn_policy = group1 + group2 + group3
        eye = torch.eye(N, dtype=attn_policy.dtype, device=attn_policy.device).view(1, 1, N, N)
        attn_policy = attn_policy + (1.0 - attn_policy) * eye

        # For stable training
        max_att, _ = torch.max(attn, dim=-1, keepdim=True)
        attn = attn - max_att
        attn = attn.to(torch.float32).exp_() * attn_policy.to(torch.float32)
        attn = (attn + eps / N) / (attn.sum(dim=-1, keepdim=True) + eps)
        return attn.type_as(max_att)

    def attn_in_group(self, q, k, v):
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(1, -1, self.dim)
        return x

    def forward(self, x, template_mask, search_feat_len, tgt_type=None, attn_masking=True, threshold=0., ratio=0.):
        B, N, C = x.shape
        decision = None
        assert not (tgt_type is None and self.early), 'conflict in implementation'

        if self.divide:
            if tgt_type == 'allmax':
                tgt_rep = x[:, :-search_feat_len]
                tgt_rep = F.adaptive_max_pool1d(tgt_rep.transpose(1, 2), output_size=1).transpose(1, 2)
            elif tgt_type == 'allavg':
                tgt_rep = x[:, :-search_feat_len]
                tgt_rep = F.adaptive_avg_pool1d(tgt_rep.transpose(1, 2), output_size=1).transpose(1, 2)
            elif tgt_type == 'roimax':
                tgt_rep = x[:, :-search_feat_len] * template_mask.unsqueeze(-1)
                tgt_rep = F.adaptive_max_pool1d(tgt_rep.transpose(1, 2), output_size=1).transpose(1, 2)
            elif tgt_type == 'roiavg':
                tgt_rep = x[:, :-search_feat_len] * template_mask.unsqueeze(-1)
                tgt_rep = F.adaptive_avg_pool1d(tgt_rep.transpose(1, 2), output_size=1).transpose(1, 2)
            else:
                raise NotImplementedError
            if self.early:
                tgt_rep = tgt_rep.expand(-1, search_feat_len, -1)
                divide_prediction = self.divide_predict(torch.cat((x[:, -search_feat_len:], tgt_rep), dim=-1))
            else:
                local_transforms = self.divide_local_transform(x[:, -search_feat_len:])
                if tgt_type is None:
                    divide_prediction = self.divide_predict(local_transforms)
                else:
                    global_transforms = self.divide_global_transform(tgt_rep)
                    divide_prediction = self.divide_predict(global_transforms + local_transforms)

            if self.gauss:
                # Smooth the selection in local neighborhood
                size = int(search_feat_len ** 0.5)
                divide_prediction = self.divide_gaussian_filter(
                    divide_prediction.transpose(1, 2).reshape(B, 2, size, size))
                divide_prediction = F.log_softmax(divide_prediction.reshape(B, 2, -1).transpose(1, 2), dim=-1)

            if self.training:
                # During training
                decision = F.gumbel_softmax(divide_prediction, hard=True)
            else:
                # During inference
                if threshold:
                    # Manual rank based selection
                    decision_rank = (F.softmax(divide_prediction, dim=-1)[:, :, 0] < threshold).long()
                else:
                    # Auto rank based selection
                    decision_rank = torch.argsort(divide_prediction, dim=-1, descending=True)[:, :, 0]

                decision = F.one_hot(decision_rank, num_classes=2)

                if ratio:
                    # Ratio based selection
                    K = int(search_feat_len * ratio)
                    _, indices = torch.topk(divide_prediction[:, :, 1], k=K, dim=-1)
                    force_back = torch.zeros(B, K, dtype=decision.dtype, device=decision.device)
                    force_over = torch.ones(B, K, dtype=decision.dtype, device=decision.device)
                    decision[:, :, 0] = torch.scatter(decision[:, :, 0], -1, indices, force_back)
                    decision[:, :, 1] = torch.scatter(decision[:, :, 1], -1, indices, force_over)

            blank_policy = torch.zeros(B, search_feat_len, 1, dtype=divide_prediction.dtype,
                                       device=divide_prediction.device)
            template_policy = torch.zeros(B, N - search_feat_len, 3, dtype=divide_prediction.dtype,
                                          device=divide_prediction.device)
            template_policy[:, :, 0] = 1
            policy = torch.cat([template_policy, torch.cat([blank_policy, decision], dim=-1)], dim=1)

            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]  # Make torchscript happy (cannot use tensor as tuple)

            if not attn_masking and not self.training:
                # Conduct three categories separately
                num_group1 = policy[:, :, 0].sum()
                num_group2 = policy[:, :, 1].sum()
                num_group3 = policy[:, :, 2].sum()
                _, E_T_ind = torch.topk(policy[:, :, 0], k=int(num_group1.item()), sorted=False)
                _, E_S_ind = torch.topk(policy[:, :, 1], k=int(num_group2.item()), sorted=False)
                _, E_A_ind = torch.topk(policy[:, :, 2], k=int(num_group3.item()), sorted=False)
                E_T_indices = E_T_ind.unsqueeze(1).unsqueeze(-1).repeat(1, self.num_heads, 1, self.head_dim)
                E_S_indices = E_S_ind.unsqueeze(1).unsqueeze(-1).repeat(1, self.num_heads, 1, self.head_dim)
                E_A_indices = E_A_ind.unsqueeze(1).unsqueeze(-1).repeat(1, self.num_heads, 1, self.head_dim)
                E_T_q = torch.gather(q, 2, E_T_indices)
                E_S_q = torch.gather(q, 2, E_S_indices)
                E_A_q = torch.gather(q, 2, E_A_indices)
                E_T_k = torch.gather(k, 2, torch.cat((E_T_indices, E_A_indices), dim=2))
                E_S_k = torch.gather(k, 2, torch.cat((E_S_indices, E_A_indices), dim=2))
                E_A_k = k
                E_T_v = torch.gather(v, 2, torch.cat((E_T_indices, E_A_indices), dim=2))
                E_S_v = torch.gather(v, 2, torch.cat((E_S_indices, E_A_indices), dim=2))
                E_A_v = v
                E_T_output = self.attn_in_group(E_T_q, E_T_k, E_T_v)
                E_S_output = self.attn_in_group(E_S_q, E_S_k, E_S_v)
                E_A_output = self.attn_in_group(E_A_q, E_A_k, E_A_v)

                x = torch.zeros_like(x, dtype=x.dtype, device=x.device)
                x = torch.scatter(x, 1, E_T_ind.unsqueeze(-1).repeat(1, 1, self.dim), E_T_output)
                x = torch.scatter(x, 1, E_S_ind.unsqueeze(-1).repeat(1, 1, self.dim), E_S_output)
                x = torch.scatter(x, 1, E_A_ind.unsqueeze(-1).repeat(1, 1, self.dim), E_A_output)
                x = self.proj(x)
            else:
                # Conduct three categories together
                attn = (q @ k.transpose(-2, -1)) * self.scale
                attn = self.softmax_with_policy(attn, policy)
                attn = self.attn_drop(attn)
                x = (attn @ v).transpose(1, 2).reshape(B, N, C)
                x = self.proj(x)
                x = self.proj_drop(x)
        else:
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]  # Make torchscript happy (cannot use tensor as tuple)

            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
            x = self.proj(x)
            x = self.proj_drop(x)
        return x, decision


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, divide=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop,
                              divide=divide)
        # Note: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, template_mask, search_feat_len, threshold, tgt_type):
        feat, decision = self.attn(self.norm1(x), template_mask, search_feat_len,
                                   threshold=threshold, tgt_type=tgt_type)
        x = x + self.drop_path(feat)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x, decision


class VisionTransformer(BaseBackbone):
    """
    Vision Transformer.
    A PyTorch impl of : 'An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale'.
        (https://arxiv.org/abs/2010.11929)
    Includes distillation token & head support for 'DeiT: Data-efficient Image Transformers'.
        (https://arxiv.org/abs/2012.12877)
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None, weight_init=''):
        """
        Args:
            img_size (int, tuple): Input image size.
            patch_size (int, tuple): Patch size.
            in_chans (int): Number of input channels.
            num_classes (int): Number of classes for classification head.
            embed_dim (int): Embedding dimension.
            depth (int): Depth of transformer.
            num_heads (int): Number of attention heads.
            mlp_ratio (int): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): Enable bias for qkv if True.
            representation_size (Optional[int]): Enable and set representation layer (pre-logits) to this value if set.
            distilled (bool): Model includes a distillation token and head as in DeiT models.
            drop_rate (float): Dropout rate.
            attn_drop_rate (float): Attention dropout rate.
            drop_path_rate (float): Stochastic depth rate.
            embed_layer (nn.Module): Patch embedding layer.
            norm_layer: (nn.Module): Normalization layer.
            weight_init: (str): Weight init scheme.
        """

        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # Stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer,
                divide=bool(i))
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        self.init_weights(weight_init)

    def init_weights(self, mode=''):
        assert mode in ('jax', 'jax_nlhb', 'nlhb', '')
        head_bias = -math.log(self.num_classes) if 'nlhb' in mode else 0.
        trunc_normal_(self.pos_embed, std=.02)
        if self.dist_token is not None:
            trunc_normal_(self.dist_token, std=.02)
        if mode.startswith('jax'):
            # Leave cls token as zeros to match jax impl
            named_apply(partial(_init_vit_weights, head_bias=head_bias, jax_impl=True), self)
        else:
            trunc_normal_(self.cls_token, std=.02)
            self.apply(_init_vit_weights)

    def _init_weights(self, m):
        # This fn left here for compat with downstream users
        _init_vit_weights(m)

    @torch.jit.ignore()
    def load_pretrained(self, checkpoint_path, prefix=''):
        _load_weights(self, checkpoint_path, prefix)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'dist_token'}

    def get_classifier(self):
        if self.dist_token is None:
            return self.head
        else:
            return self.head, self.head_dist

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        if self.num_tokens == 2:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()


def _init_vit_weights(module: nn.Module, name: str = '', head_bias: float = 0., jax_impl: bool = False):
    """
    ViT weight initialization.

    When called without n, head_bias, jax_impl args it will behave exactly the same
    as my original init for compatibility with prev hparam / downstream use cases (ie DeiT).
    When called w/ valid n (module name) and jax_impl=True, will (hopefully) match JAX impl.
    """

    if isinstance(module, nn.Linear):
        if name.startswith('head'):
            nn.init.zeros_(module.weight)
            nn.init.constant_(module.bias, head_bias)
        elif name.startswith('pre_logits'):
            lecun_normal_(module.weight)
            nn.init.zeros_(module.bias)
        else:
            if jax_impl:
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    if 'mlp' in name:
                        nn.init.normal_(module.bias, std=1e-6)
                    else:
                        nn.init.zeros_(module.bias)
            else:
                trunc_normal_(module.weight, std=.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    elif jax_impl and isinstance(module, nn.Conv2d):
        # Note: conv was left to pytorch default in my original init
        lecun_normal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
        nn.init.zeros_(module.bias)
        nn.init.ones_(module.weight)


@torch.no_grad()
def _load_weights(model: VisionTransformer, checkpoint_path: str, prefix: str = ''):
    """
    Load weights from .npz checkpoints for official Google Brain Flax implementation.
    """

    import numpy as np

    def _n2p(w, t=True):
        if w.ndim == 4 and w.shape[0] == w.shape[1] == w.shape[2] == 1:
            w = w.flatten()
        if t:
            if w.ndim == 4:
                w = w.transpose([3, 2, 0, 1])
            elif w.ndim == 3:
                w = w.transpose([2, 0, 1])
            elif w.ndim == 2:
                w = w.transpose([1, 0])
        return torch.from_numpy(w)

    w = np.load(checkpoint_path)
    if not prefix and 'opt/target/embedding/kernel' in w:
        prefix = 'opt/target/'

    if hasattr(model.patch_embed, 'backbone'):
        # Hybrid
        backbone = model.patch_embed.backbone
        stem_only = not hasattr(backbone, 'stem')
        stem = backbone if stem_only else backbone.stem
        stem.conv.weight.copy_(adapt_input_conv(stem.conv.weight.shape[1], _n2p(w[f'{prefix}conv_root/kernel'])))
        stem.norm.weight.copy_(_n2p(w[f'{prefix}gn_root/scale']))
        stem.norm.bias.copy_(_n2p(w[f'{prefix}gn_root/bias']))
        if not stem_only:
            for i, stage in enumerate(backbone.stages):
                for j, block in enumerate(stage.blocks):
                    bp = f'{prefix}block{i + 1}/unit{j + 1}/'
                    for r in range(3):
                        getattr(block, f'conv{r + 1}').weight.copy_(_n2p(w[f'{bp}conv{r + 1}/kernel']))
                        getattr(block, f'norm{r + 1}').weight.copy_(_n2p(w[f'{bp}gn{r + 1}/scale']))
                        getattr(block, f'norm{r + 1}').bias.copy_(_n2p(w[f'{bp}gn{r + 1}/bias']))
                    if block.downsample is not None:
                        block.downsample.conv.weight.copy_(_n2p(w[f'{bp}conv_proj/kernel']))
                        block.downsample.norm.weight.copy_(_n2p(w[f'{bp}gn_proj/scale']))
                        block.downsample.norm.bias.copy_(_n2p(w[f'{bp}gn_proj/bias']))
        embed_conv_w = _n2p(w[f'{prefix}embedding/kernel'])
    else:
        embed_conv_w = adapt_input_conv(
            model.patch_embed.proj.weight.shape[1], _n2p(w[f'{prefix}embedding/kernel']))
    model.patch_embed.proj.weight.copy_(embed_conv_w)
    model.patch_embed.proj.bias.copy_(_n2p(w[f'{prefix}embedding/bias']))
    model.cls_token.copy_(_n2p(w[f'{prefix}cls'], t=False))
    pos_embed_w = _n2p(w[f'{prefix}Transformer/posembed_input/pos_embedding'], t=False)
    if pos_embed_w.shape != model.pos_embed.shape:
        pos_embed_w = resize_pos_embed(  # Resize pos embedding when different size from pretrained weights
            pos_embed_w, model.pos_embed, getattr(model, 'num_tokens', 1), model.patch_embed.grid_size)
    model.pos_embed.copy_(pos_embed_w)
    model.norm.weight.copy_(_n2p(w[f'{prefix}Transformer/encoder_norm/scale']))
    model.norm.bias.copy_(_n2p(w[f'{prefix}Transformer/encoder_norm/bias']))
    if isinstance(model.head, nn.Linear) and model.head.bias.shape[0] == w[f'{prefix}head/bias'].shape[-1]:
        model.head.weight.copy_(_n2p(w[f'{prefix}head/kernel']))
        model.head.bias.copy_(_n2p(w[f'{prefix}head/bias']))
    if isinstance(getattr(model.pre_logits, 'fc', None), nn.Linear) and f'{prefix}pre_logits/bias' in w:
        model.pre_logits.fc.weight.copy_(_n2p(w[f'{prefix}pre_logits/kernel']))
        model.pre_logits.fc.bias.copy_(_n2p(w[f'{prefix}pre_logits/bias']))
    for i, block in enumerate(model.blocks.children()):
        block_prefix = f'{prefix}Transformer/encoderblock_{i}/'
        mha_prefix = block_prefix + 'MultiHeadDotProductAttention_1/'
        block.norm1.weight.copy_(_n2p(w[f'{block_prefix}LayerNorm_0/scale']))
        block.norm1.bias.copy_(_n2p(w[f'{block_prefix}LayerNorm_0/bias']))
        block.attn.qkv.weight.copy_(torch.cat([
            _n2p(w[f'{mha_prefix}{n}/kernel'], t=False).flatten(1).T for n in ('query', 'key', 'value')]))
        block.attn.qkv.bias.copy_(torch.cat([
            _n2p(w[f'{mha_prefix}{n}/bias'], t=False).reshape(-1) for n in ('query', 'key', 'value')]))
        block.attn.proj.weight.copy_(_n2p(w[f'{mha_prefix}out/kernel']).flatten(1))
        block.attn.proj.bias.copy_(_n2p(w[f'{mha_prefix}out/bias']))
        for r in range(2):
            getattr(block.mlp, f'fc{r + 1}').weight.copy_(_n2p(w[f'{block_prefix}MlpBlock_3/Dense_{r}/kernel']))
            getattr(block.mlp, f'fc{r + 1}').bias.copy_(_n2p(w[f'{block_prefix}MlpBlock_3/Dense_{r}/bias']))
        block.norm2.weight.copy_(_n2p(w[f'{block_prefix}LayerNorm_2/scale']))
        block.norm2.bias.copy_(_n2p(w[f'{block_prefix}LayerNorm_2/bias']))


def resize_pos_embed(posemb, posemb_new, num_tokens=1, gs_new=()):
    # Rescale the grid of position embeddings when loading from state_dict. Adapted from
    # https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224
    print('resized position embedding %s to %s', posemb.shape, posemb_new.shape)
    ntok_new = posemb_new.shape[1]
    if num_tokens:
        posemb_tok, posemb_grid = posemb[:, :num_tokens], posemb[0, num_tokens:]
        ntok_new -= num_tokens
    else:
        posemb_tok, posemb_grid = posemb[:, :0], posemb[0]
    gs_old = int(math.sqrt(len(posemb_grid)))
    if not len(gs_new):  # Backwards compatibility
        gs_new = [int(math.sqrt(ntok_new))] * 2
    assert len(gs_new) >= 2
    print('position embedding grid-size from %s to %s', [gs_old, gs_old], gs_new)
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=gs_new, mode='bilinear')
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_new[0] * gs_new[1], -1)
    posemb = torch.cat([posemb_tok, posemb_grid], dim=1)
    return posemb


def checkpoint_filter_fn(state_dict, model):
    """
    Convert patch embedding weight from manual patchify + linear proj to conv.
    """

    out_dict = dict()
    if 'model' in state_dict:
        # For DeiT models
        state_dict = state_dict['model']
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k and len(v.shape) < 4:
            # For old models that I trained prior to conv based patchification
            O, I, H, W = model.patch_embed.proj.weight.shape
            v = v.reshape(O, -1, H, W)
        elif k == 'pos_embed' and v.shape != model.pos_embed.shape:
            # To resize pos embedding when using model at different size from pretrained weights
            v = resize_pos_embed(
                v, model.pos_embed, getattr(model, 'num_tokens', 1), model.patch_embed.grid_size)
        out_dict[k] = v
    return out_dict


def _create_vision_transformer(variant, pretrained=False, default_cfg=None, **kwargs):
    if kwargs.get('features_only', None):
        raise RuntimeError('ERROR: features_only not implemented for Vision Transformer models')

    model = VisionTransformer(**kwargs)

    if pretrained:
        if 'npz' in pretrained:
            model.load_pretrained(pretrained, prefix='')
        else:
            checkpoint = torch.load(pretrained, map_location='cpu')
            missing_keys, unexpected_keys = model.load_state_dict(checkpoint['model'], strict=False)
            print('load pretrained model from ' + pretrained)
    return model


def vit_base_patch16_224_base(pretrained=False, **kwargs):
    """
    ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    """

    model_kwargs = dict(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer('vit_base_patch16_224_in21k', pretrained=pretrained, **model_kwargs)
    return model


def vit_base_patch16_224_large(pretrained=False, **kwargs):
    """
    ViT-Large model (ViT-L/16) from original paper (https://arxiv.org/abs/2010.11929).
    """

    model_kwargs = dict(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, **kwargs)
    model = _create_vision_transformer('vit_large_patch16_224_in21k', pretrained=pretrained, **model_kwargs)
    return model
