import torch
import torch.nn as nn
from timm.models.layers import to_2tuple

from lib.models.layers.patch_embed import PatchEmbed
from lib.models.grm.utils import combine_tokens, recover_tokens


class BaseBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        # For original ViT
        self.pos_embed = None
        self.img_size = [224, 224]
        self.patch_size = 16
        self.embed_dim = 384

        self.cat_mode = 'direct'

        self.pos_embed_x = None
        self.pos_embed_z = None

    def finetune_track(self, cfg, patch_start_index=1):
        search_size = to_2tuple(cfg.DATA.SEARCH.SIZE)
        template_size = to_2tuple(cfg.DATA.TEMPLATE.SIZE)
        new_patch_size = cfg.MODEL.BACKBONE.STRIDE

        self.cat_mode = cfg.MODEL.BACKBONE.CAT_MODE

        # Resize patch embedding
        if new_patch_size != self.patch_size:
            print('inconsistent patch size with the pretrained weights, interpolate the weight')
            old_patch_embed = dict()
            for name, param in self.patch_embed.named_parameters():
                if 'weight' in name:
                    param = nn.functional.interpolate(param, size=(new_patch_size, new_patch_size),
                                                      mode='bicubic', align_corners=False)
                    param = nn.Parameter(param)
                old_patch_embed[name] = param
            self.patch_embed = PatchEmbed(img_size=self.img_size, patch_size=new_patch_size, in_chans=3,
                                          embed_dim=self.embed_dim)
            self.patch_embed.proj.bias = old_patch_embed['proj.bias']
            self.patch_embed.proj.weight = old_patch_embed['proj.weight']

        # For patch embedding
        patch_pos_embed = self.pos_embed[:, patch_start_index:, :]
        patch_pos_embed = patch_pos_embed.transpose(1, 2)
        B, E, Q = patch_pos_embed.shape
        P_H, P_W = self.img_size[0] // self.patch_size, self.img_size[1] // self.patch_size
        patch_pos_embed = patch_pos_embed.view(B, E, P_H, P_W)

        # For search region
        H, W = search_size
        new_P_H, new_P_W = H // new_patch_size, W // new_patch_size
        search_patch_pos_embed = nn.functional.interpolate(patch_pos_embed, size=(new_P_H, new_P_W), mode='bicubic',
                                                           align_corners=False)
        search_patch_pos_embed = search_patch_pos_embed.flatten(2).transpose(1, 2)

        # For template region
        H, W = template_size
        new_P_H, new_P_W = H // new_patch_size, W // new_patch_size
        template_patch_pos_embed = nn.functional.interpolate(patch_pos_embed, size=(new_P_H, new_P_W), mode='bicubic',
                                                             align_corners=False)
        template_patch_pos_embed = template_patch_pos_embed.flatten(2).transpose(1, 2)

        self.pos_embed_x = nn.Parameter(search_patch_pos_embed)
        self.pos_embed_z = nn.Parameter(template_patch_pos_embed)

    def forward_features(self, z, x, template_mask, search_feat_len, threshold, tgt_type):
        x = self.patch_embed(x)
        z = self.patch_embed(z)

        x += self.pos_embed_x
        z += self.pos_embed_z

        x = combine_tokens(z, x, mode=self.cat_mode)

        x = self.pos_drop(x)

        decisions = list()
        for i, blk in enumerate(self.blocks):
            x, decision = blk(x, template_mask, search_feat_len, threshold=threshold, tgt_type=tgt_type)
            if decision is not None and self.training:
                map_size = decision.shape[1]
                decision = decision[:, :, -1].sum(dim=-1, keepdim=True) / map_size
                decisions.append(decision)

        x = recover_tokens(x, mode=self.cat_mode)

        if self.training:
            decisions = torch.cat(decisions, dim=-1)  # .mean(dim=-1, keepdim=True)
        return self.norm(x), decisions

    def forward(self, z, x, template_mask, search_feat_len, threshold, tgt_type, **kwargs):
        """
        Joint feature extraction and relation modeling for the basic ViT backbone.

        Args:
            z (torch.Tensor): Template feature, [B, C, H_z, W_z].
            x (torch.Tensor): Search region feature, [B, C, H_x, W_x].

        Returns:
            x (torch.Tensor): Merged template and search region feature, [B, L_z+L_x, C].
            attn : None.
        """

        x, decisions = self.forward_features(z, x, template_mask, search_feat_len, threshold, tgt_type)
        return x, decisions
