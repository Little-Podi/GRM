import os


class EnvironmentSettings:
    def __init__(self):
        self.workspace_dir = os.path.expanduser(
            '~') + '/track/code/GRM'  # Base directory for saving network checkpoints
        self.tensorboard_dir = os.path.expanduser(
            '~') + '/track/code/GRM/tensorboard'  # Directory for tensorboard files
        self.pretrained_networks = os.path.expanduser('~') + '/track/code/GRM/pretrained_networks'
        self.lasot_dir = os.path.expanduser('~') + '/track/data/LaSOT'
        self.got10k_dir = os.path.expanduser('~') + '/track/data/GOT10k/train'
        self.got10k_val_dir = os.path.expanduser('~') + '/track/data/GOT10k/val'
        self.trackingnet_dir = os.path.expanduser('~') + '/track/data/TrackingNet'
        self.coco_dir = os.path.expanduser('~') + '/track/data/COCO'
        self.lvis_dir = ''
        self.sbd_dir = ''
        self.imagenet_dir = ''
        self.imagenetdet_dir = ''
        self.ecssd_dir = ''
        self.hkuis_dir = ''
        self.msra10k_dir = ''
        self.davis_dir = ''
        self.youtubevos_dir = ''
