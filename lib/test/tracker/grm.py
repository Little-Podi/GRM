import os

# For debug
import cv2
import torch

from lib.models.grm import build_grm
from lib.test.tracker.basetracker import BaseTracker
from lib.test.tracker.data_utils import Preprocessor
from lib.test.utils.hann import hann2d
from lib.train.data.processing_utils import sample_target
from lib.utils.box_ops import clip_box
from lib.utils.mask_utils import generate_mask_cond


class GRM(BaseTracker):
    def __init__(self, params, dataset_name):
        super(GRM, self).__init__(params)
        network = build_grm(params.cfg, training=False)
        network.load_state_dict(torch.load(self.params.checkpoint, map_location='cpu')['net'], strict=True)
        self.cfg = params.cfg
        self.network = network.cuda()
        self.network.eval()
        self.preprocessor = Preprocessor()
        self.state = None
        self.feat_sz = self.cfg.TEST.SEARCH_SIZE // self.cfg.MODEL.BACKBONE.STRIDE

        # Motion constrain
        self.output_window = hann2d(torch.tensor([self.feat_sz, self.feat_sz]).long(), centered=True).cuda()

        # For debug
        self.debug = params.debug
        self.use_visdom = params.debug
        self.frame_id = 0
        if self.debug:
            if not self.use_visdom:
                self.save_dir = 'debug'
                if not os.path.exists(self.save_dir):
                    os.makedirs(self.save_dir)
            else:
                # self.add_hook()
                self._init_visdom(None, 1)

        # For save boxes from all queries
        self.save_all_boxes = params.save_all_boxes

        # Set the hyper-parameters
        DATASET_NAME = dataset_name.upper()
        if hasattr(self.cfg.TEST.HYPER, DATASET_NAME):
            self.threshold = self.cfg.TEST.HYPER[DATASET_NAME]
        else:
            self.threshold = self.cfg.TEST.HYPER.DEFAULT

    def initialize(self, image, info: dict):
        # Forward the template once
        z_patch_arr, resize_factor, z_amask_arr = sample_target(image, info['init_bbox'], self.params.template_factor,
                                                                output_sz=self.params.template_size)
        template = self.preprocessor.process(z_patch_arr, z_amask_arr)
        self.z_patch_arr = z_patch_arr
        self.z_dict1 = template

        template_bbox = self.transform_bbox_to_crop(info['init_bbox'], resize_factor,
                                                    template.tensors.device).squeeze(1)
        self.box_mask_z = generate_mask_cond(self.cfg, 1, template.tensors.device, template_bbox)

        # Save states
        self.state = info['init_bbox']
        self.frame_id = 0
        if self.save_all_boxes:
            # Save all predicted boxes
            all_boxes_save = info['init_bbox'] * self.cfg.MODEL.NUM_OBJECT_QUERIES
            return {'all_boxes': all_boxes_save}

    def track(self, image, info: dict = None):
        H, W, _ = image.shape
        self.frame_id += 1
        x_patch_arr, resize_factor, x_amask_arr = sample_target(image, self.state, self.params.search_factor,
                                                                output_sz=self.params.search_size)  # (x1, y1, w, h)
        search = self.preprocessor.process(x_patch_arr, x_amask_arr)

        with torch.no_grad():
            x_dict = search
            # Merge the template and the search
            # Run the transformer
            out_dict = self.network.forward(template=self.z_dict1.tensors, search=x_dict.tensors,
                                            template_mask=self.box_mask_z, threshold=self.threshold)

        # Add hann windows
        pred_score_map = out_dict['score_map']
        response = self.output_window * pred_score_map
        pred_boxes = self.network.box_head.cal_bbox(response, out_dict['size_map'], out_dict['offset_map'])
        pred_boxes = pred_boxes.view(-1, 4)
        # Baseline: Take the mean of all pred boxes as the final result
        pred_box = (pred_boxes.mean(dim=0) * self.params.search_size / resize_factor).tolist()  # (cx, cy, w, h) [0,1]
        # Get the final box result
        self.state = clip_box(self.map_box_back(pred_box, resize_factor), H, W, margin=10)

        # For debug
        if self.debug:
            if not self.use_visdom:
                x1, y1, w, h = self.state
                image_BGR = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                cv2.rectangle(image_BGR, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color=(0, 0, 255), thickness=2)
                save_path = os.path.join(self.save_dir, '%04d.jpg' % self.frame_id)
                cv2.imwrite(save_path, image_BGR)

        if self.save_all_boxes:
            # Save all predictions
            all_boxes = self.map_box_back_batch(pred_boxes * self.params.search_size / resize_factor, resize_factor)
            all_boxes_save = all_boxes.view(-1).tolist()  # (4N, )
            return {'target_bbox': self.state,
                    'all_boxes': all_boxes_save}
        else:
            return {'target_bbox': self.state}

    def map_box_back(self, pred_box: list, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]

    def map_box_back_batch(self, pred_box: torch.Tensor, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box.unbind(-1)  # (N,4) --> (N,)
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return torch.stack([cx_real - 0.5 * w, cy_real - 0.5 * h, w, h], dim=-1)

    def add_hook(self):
        conv_features, enc_attn_weights, dec_attn_weights = list(), list(), list()

        for i in range(12):
            self.network.backbone.blocks[i].attn.register_forward_hook(
                lambda self, input, output: enc_attn_weights.append(output[1])
            )

        self.enc_attn_weights = enc_attn_weights


def get_tracker_class():
    return GRM
