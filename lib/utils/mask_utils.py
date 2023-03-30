import torch


def generate_bbox_mask(bbox_mask, bbox):
    b, h, w = bbox_mask.shape
    for i in range(b):
        bbox_i = bbox[i].cpu().tolist()
        bbox_mask[i, max(int(bbox_i[1]), 0):min(int(bbox_i[1] + bbox_i[3] + 1), h),
        max(int(bbox_i[0]), 0):min(int(bbox_i[0] + bbox_i[2] + 1), w)] = 1
    return bbox_mask


def generate_mask_cond(cfg, bs, device, gt_bbox):
    template_size = cfg.DATA.TEMPLATE.SIZE
    stride = cfg.MODEL.BACKBONE.STRIDE
    feat_size = template_size // stride

    box_mask_z = torch.zeros([bs, feat_size, feat_size], device=device)
    box_mask_z = generate_bbox_mask(box_mask_z, gt_bbox * feat_size).unsqueeze(1).to(torch.float).flatten(1)
    return box_mask_z
