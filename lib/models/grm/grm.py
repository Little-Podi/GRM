"""
Basic GRM model.
"""

import os

import torch
from torch import nn

from lib.models.layers.head import build_box_head
from lib.models.grm.vit import vit_base_patch16_224_base, vit_base_patch16_224_large
from lib.utils.box_ops import box_xyxy_to_cxcywh


class GRM(nn.Module):
    """
    This is the base class for GRM.
    """

    def __init__(self, transformer, box_head, head_type='CORNER', tgt_type='allmax'):
        """
        Initializes the model.

        Parameters:
            transformer: Torch module of the transformer architecture.
        """

        super().__init__()
        self.backbone = transformer
        self.box_head = box_head
        self.head_type = head_type
        if head_type == 'CORNER' or head_type == 'CENTER':
            self.feat_sz_s = int(box_head.feat_sz)
            self.feat_len_s = int(box_head.feat_sz ** 2)
        self.tgt_type = tgt_type

    def forward(self, template: torch.Tensor, search: torch.Tensor, template_mask=None, threshold=0.):
        x, decisions = self.backbone(z=template, x=search, template_mask=template_mask, search_feat_len=self.feat_len_s,
                                     threshold=threshold, tgt_type=self.tgt_type)

        # Forward head
        feat_last = x
        if isinstance(x, list):
            feat_last = x[-1]
        out = self.forward_head(feat_last, None)
        out['decisions'] = decisions
        return out

    def forward_head(self, cat_feature, gt_score_map=None):
        """
        cat_feature: Output embeddings of the backbone, it can be (HW1+HW2, B, C) or (HW2, B, C).
        """

        enc_opt = cat_feature[:, -self.feat_len_s:]  # Encoder output for the search region (B, HW, C)
        opt = (enc_opt.unsqueeze(-1)).permute((0, 3, 2, 1)).contiguous()
        bs, Nq, C, HW = opt.size()
        opt_feat = opt.view(-1, C, self.feat_sz_s, self.feat_sz_s)

        if self.head_type == 'CORNER':
            # Run the corner head
            pred_box, score_map = self.box_head(opt_feat, True)
            outputs_coord = box_xyxy_to_cxcywh(pred_box)
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map}
            return out
        elif self.head_type == 'CENTER':
            # Run the center head
            score_map_ctr, bbox, size_map, offset_map = self.box_head(opt_feat, gt_score_map)
            outputs_coord = bbox
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map_ctr,
                   'size_map': size_map,
                   'offset_map': offset_map}
            return out
        else:
            raise NotImplementedError


def build_grm(cfg, training=True):
    current_dir = os.path.dirname(os.path.abspath(__file__))  # This is your Project Root
    pretrained_path = os.path.join(current_dir, '../../../pretrained_models')
    if cfg.MODEL.PRETRAIN_FILE and ('GRM' not in cfg.MODEL.PRETRAIN_FILE) and training:
        pretrained = os.path.join(pretrained_path, cfg.MODEL.PRETRAIN_FILE)
    else:
        pretrained = ''

    if cfg.MODEL.BACKBONE.TYPE == 'vit_base':
        if cfg.MODEL.PRETRAIN_FILE == 'mae_pretrain_vit_base.pth':
            backbone = vit_base_patch16_224_base(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE)
            hidden_dim = backbone.embed_dim
            patch_start_index = 1
        elif cfg.MODEL.PRETRAIN_FILE == 'deit_base_patch16_224-b5f2ef4d.pth':
            backbone = vit_base_patch16_224_base(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE)
            hidden_dim = backbone.embed_dim
            patch_start_index = 1
        elif cfg.MODEL.PRETRAIN_FILE == 'deit_base_distilled_patch16_224-df68dfff.pth':
            backbone = vit_base_patch16_224_base(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE, distilled=True)
            hidden_dim = backbone.embed_dim
            patch_start_index = 2
        else:
            raise NotImplementedError
    elif cfg.MODEL.BACKBONE.TYPE == 'vit_large':
        if cfg.MODEL.PRETRAIN_FILE == 'mae_pretrain_vit_large.pth':
            backbone = vit_base_patch16_224_large(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE)
            hidden_dim = backbone.embed_dim
            patch_start_index = 1
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

    backbone.finetune_track(cfg=cfg, patch_start_index=patch_start_index)

    box_head = build_box_head(cfg, hidden_dim)

    model = GRM(
        backbone,
        box_head,
        head_type=cfg.MODEL.HEAD.TYPE,
        tgt_type=cfg.MODEL.TGT_TYPE
    )

    if 'GRM' in cfg.MODEL.PRETRAIN_FILE and training:
        checkpoint = torch.load(cfg.MODEL.PRETRAIN_FILE, map_location='cpu')
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint['net'], strict=False)
        print('load pretrained model from ' + cfg.MODEL.PRETRAIN_FILE)
    return model
