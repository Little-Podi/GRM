import argparse
import os
import random
import warnings

warnings.filterwarnings('ignore')


def parse_args():
    """
    Args for training.
    """

    parser = argparse.ArgumentParser(description='parse args for training')
    # For train
    parser.add_argument('--script', type=str, default='grm', help='training script name')
    parser.add_argument('--config', type=str, default='vitb_256_ep300', help='yaml configure file name')
    parser.add_argument('--save_dir', type=str, default='./output',
                        help='root directory to save checkpoints, logs, and tensorboard')
    parser.add_argument('--mode', type=str, choices=['single', 'multiple'], default='single',
                        help='train on single gpu or multiple gpus')
    parser.add_argument('--nproc', type=int, default=8,
                        help='number of GPUs per node')  # Specify when mode is multiple
    parser.add_argument('--use_lmdb', type=int, choices=[0, 1], default=0)  # Whether datasets are in lmdb format
    parser.add_argument('--script_prv', type=str, help='training script name')
    parser.add_argument('--config_prv', type=str, default='baseline', help='yaml configure file name')
    parser.add_argument('--use_wandb', type=int, choices=[0, 1], default=0)  # Whether to use wandb
    # For knowledge distillation
    parser.add_argument('--distill', type=int, choices=[0, 1], default=0)  # Whether to use knowledge distillation
    parser.add_argument('--script_teacher', type=str, help='teacher script name')
    parser.add_argument('--config_teacher', type=str, help='teacher yaml configure file name')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    if args.mode == 'single':
        train_cmd = 'python lib/train/run_training.py --script %s --config %s --save_dir %s --use_lmdb %d ' \
                    '--script_prv %s --config_prv %s --distill %d --script_teacher %s --config_teacher %s --use_wandb %d' \
                    % (args.script, args.config, args.save_dir, args.use_lmdb, args.script_prv, args.config_prv,
                       args.distill, args.script_teacher, args.config_teacher, args.use_wandb)
    elif args.mode == 'multiple':
        train_cmd = 'python -m torch.distributed.launch --nproc_per_node %d --master_port %d lib/train/run_training.py ' \
                    '--script %s --config %s --save_dir %s --use_lmdb %d --script_prv %s --config_prv %s --use_wandb %d ' \
                    '--distill %d --script_teacher %s --config_teacher %s' \
                    % (args.nproc, random.randint(10000, 50000), args.script, args.config, args.save_dir,
                       args.use_lmdb, args.script_prv, args.config_prv, args.use_wandb,
                       args.distill, args.script_teacher, args.config_teacher)
    else:
        raise ValueError("ERROR: mode should be 'single' or 'multiple'")
    os.system(train_cmd)


if __name__ == '__main__':
    main()
