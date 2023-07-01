import torch

from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy
from . import BaseActor
from ...utils.heapmap_utils import generate_heatmap
from ...utils.mask_utils import generate_mask_cond


class GRMActor(BaseActor):
    """
    Actor for training GRM models.
    """

    def __init__(self, net, objective, loss_weight, settings, cfg=None):
        super().__init__(net, objective)
        self.loss_weight = loss_weight
        self.settings = settings
        self.bs = self.settings.batchsize  # Batch size
        self.cfg = cfg

    def __call__(self, data):
        """
        Args:
            data: The input data, should contain the fields 'template', 'search', 'gt_bbox'.
            template_images: (N_t, batch, 3, H, W)
            search_images: (N_s, batch, 3, H, W)

        Returns:
            loss: The training loss.
            status: Dict containing detailed losses.
        """

        # Forward pass
        out_dict = self.forward_pass(data)

        # Compute losses
        loss, status = self.compute_losses(out_dict, data)
        return loss, status

    def forward_pass(self, data):
        # Currently only support 1 template and 1 search region
        assert len(data['template_images']) == 1
        assert len(data['search_images']) == 1

        template_list = list()
        for i in range(self.settings.num_template):
            template_img_i = data['template_images'][i].view(-1,
                                                             *data['template_images'].shape[2:])  # (batch, 3, 128, 128)
            template_list.append(template_img_i)

        search_img = data['search_images'][0].view(-1, *data['search_images'].shape[2:])  # (batch, 3, 320, 320)

        box_mask_z = generate_mask_cond(self.cfg, template_list[0].shape[0], template_list[0].device,
                                        data['template_anno'][0])

        if len(template_list) == 1:
            template_list = template_list[0]

        out_dict = self.net(template=template_list, search=search_img, template_mask=box_mask_z)
        return out_dict

    def compute_losses(self, pred_dict, gt_dict, return_status=True, entropy=False):
        # GT gaussian map
        gt_bbox = gt_dict['search_anno'][-1]  # (Ns, batch, 4) (x1,y1,w,h) -> (batch, 4)
        gt_gaussian_maps = generate_heatmap(gt_dict['search_anno'], self.cfg.DATA.SEARCH.SIZE,
                                            self.cfg.MODEL.BACKBONE.STRIDE)
        gt_gaussian_maps = gt_gaussian_maps[-1].unsqueeze(1)

        # Get boxes
        pred_boxes = pred_dict['pred_boxes']
        if torch.isnan(pred_boxes).any():
            raise ValueError('ERROR: network outputs is NAN! stop training')
        num_queries = pred_boxes.size(1)
        # (B,N,4) --> (BN,4) (x1,y1,x2,y2)
        pred_boxes_vec = box_cxcywh_to_xyxy(pred_boxes).view(-1, 4)
        # (B,4) --> (B,1,4) --> (B,N,4)
        gt_boxes_vec = box_xywh_to_xyxy(gt_bbox)[:, None, :].repeat((1, num_queries, 1)).view(-1, 4).clamp(min=0.0,
                                                                                                           max=1.0)
        # Compute GIoU and IoU
        try:
            giou_loss, iou = self.objective['giou'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
        except:
            giou_loss, iou = torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()
        # Compute L1 loss
        l1_loss = self.objective['l1'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
        # Compute location loss
        if 'score_map' in pred_dict:
            location_loss = self.objective['focal'](pred_dict['score_map'], gt_gaussian_maps)
        else:
            location_loss = torch.tensor(0.0, device=l1_loss.device)

        # Weighted sum
        loss = self.loss_weight['giou'] * giou_loss + self.loss_weight['l1'] * l1_loss + self.loss_weight[
            'focal'] * location_loss
        if entropy and pred_dict['decisions'] != []:
            epsilon = 1e-5
            prob1 = pred_dict['decisions']
            prob2 = 1 - pred_dict['decisions']
            entropy_loss = (1 + prob1 * torch.log2(prob1 + epsilon) + prob2 * torch.log2(prob2 + epsilon)).mean()
            loss += entropy_loss

        if return_status:
            # Status for log
            mean_iou = iou.detach().mean()
            if entropy and pred_dict['decisions'] != []:
                status = {'Ls/total': loss.item(),
                          'Ls/giou': giou_loss.item(),
                          'Ls/l1': l1_loss.item(),
                          'Ls/loc': location_loss.item(),
                          'Ls/entropy': entropy_loss.item(),
                          'IoU': mean_iou.item()}
            else:
                status = {'Ls/total': loss.item(),
                          'Ls/giou': giou_loss.item(),
                          'Ls/l1': l1_loss.item(),
                          'Ls/loc': location_loss.item(),
                          'IoU': mean_iou.item()}
            return loss, status
        else:
            return loss
