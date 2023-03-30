# For import modules
import importlib
import os

from torch.nn import BCEWithLogitsLoss
from torch.nn.functional import l1_loss
# Distributed training related
from torch.nn.parallel import DistributedDataParallel as DDP

# Network related
from lib.models.grm import build_grm
# Forward propagation related
from lib.train.actors import GRMActor
# Train pipeline related
from lib.train.trainers import LTRTrainer
# Loss function related
from lib.utils.box_ops import giou_loss
# Some more advanced functions
from .base_functions import *
from ..utils.focal_loss import FocalLoss


def run(settings):
    settings.description = 'training script'

    # Update the default configs with config file
    if not os.path.exists(settings.cfg_file):
        raise ValueError("ERROR: %s doesn't exist" % settings.cfg_file)
    config_module = importlib.import_module('lib.config.%s.config' % settings.script_name)
    cfg = config_module.cfg
    config_module.update_config_from_file(settings.cfg_file)
    # if settings.local_rank in [-1, 0]:
    #     print('new configuration is shown below')
    #     for key in cfg.keys():
    #         print('%s configuration:' % key, cfg[key])
    #         print('\n')

    # Update settings based on cfg
    update_settings(settings, cfg)

    # Record the training log
    log_dir = os.path.join(settings.save_dir, 'logs')
    if settings.local_rank in [-1, 0]:
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
    settings.log_file = os.path.join(log_dir, '%s-%s.log' % (settings.script_name, settings.config_name))

    # Build dataloaders
    loader_train = build_dataloaders(cfg, settings)

    if 'RepVGG' in cfg.MODEL.BACKBONE.TYPE or 'swin' in cfg.MODEL.BACKBONE.TYPE or 'LightTrack' in cfg.MODEL.BACKBONE.TYPE:
        cfg.ckpt_dir = settings.save_dir

    # Create network
    if 'grm' in settings.script_name:
        net = build_grm(cfg)
    else:
        raise ValueError('ERROR: illegal script name')

    # Wrap networks to distributed one
    net.cuda()
    if settings.local_rank != -1:
        # net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)  # add syncBN converter
        net = DDP(net, device_ids=[settings.local_rank], find_unused_parameters=True)
        settings.device = torch.device('cuda:%d' % settings.local_rank)
    else:
        settings.device = torch.device('cuda:0')
    settings.deep_sup = getattr(cfg.TRAIN, 'DEEP_SUPERVISION', False)
    settings.distill = getattr(cfg.TRAIN, 'DISTILL', False)
    settings.distill_loss_type = getattr(cfg.TRAIN, 'DISTILL_LOSS_TYPE', 'KL')
    # Loss functions and actors
    if 'grm' in settings.script_name:
        focal_loss = FocalLoss()
        objective = {'giou': giou_loss, 'l1': l1_loss, 'focal': focal_loss, 'cls': BCEWithLogitsLoss()}
        loss_weight = {'giou': cfg.TRAIN.GIOU_WEIGHT, 'l1': cfg.TRAIN.L1_WEIGHT, 'focal': 1., 'cls': 1.}
        actor = GRMActor(net=net, objective=objective, loss_weight=loss_weight, settings=settings, cfg=cfg)
    else:
        raise ValueError('ERROR: illegal script name')

    # Optimizer, parameters, and learning rates
    optimizer, lr_scheduler = get_optimizer_scheduler(net, cfg)
    trainer = LTRTrainer(actor, [loader_train], optimizer, settings, lr_scheduler)

    # Train process
    trainer.train(cfg.TRAIN.EPOCH, load_latest=True, fail_safe=True)
