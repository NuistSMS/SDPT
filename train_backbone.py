import argparse
import logging
import os
import random
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
# from networks.segformer import MySegFormer as ViT_seg
from networks.MISSFOREMR import MISSFormer
from trainer_backbone import trainer_brats19
import warnings
import yaml


def save_args_to_config(args, save_path):
    """
    Save parsed command-line arguments as a YAML configuration file.

    Args:
        args: Parsed ``argparse.Namespace`` object.
        save_path: Destination YAML path.
    """
    # Convert the Namespace to a serializable dictionary.
    args_dict = vars(args)

    # Create the destination directory when needed.
    save_dir = os.path.dirname(save_path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Write a human-readable YAML file.
    with open(save_path, 'w', encoding='utf-8') as f:
        yaml.dump(
            args_dict,
            f,
            indent=4,
            sort_keys=False,
            allow_unicode=True
        )
    print(f"Training configuration saved to: {save_path}")


warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description='Initialize the shared SDPT backbone on processed BraTS data')
parser.add_argument('--root_path', type=str, required=True, help='root directory for processed data')
parser.add_argument('--dataset', type=str,
                    default='Brats19', help='experiment_name')
parser.add_argument('--list_dir', type=str, required=True, help='split-list directory')
parser.add_argument('--train_list', type=str, required=True, help='training split filename')
parser.add_argument('--val_list', type=str, required=True, help='validation split filename')
parser.add_argument('--num_classes', type=int,
                    default=4, help='output channel of network')
parser.add_argument('--output_dir', type=str, required=True, help='output directory')
# parser.add_argument('--max_iterations', type=int,
#                     default=90000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int,
                    default=65, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int,
                    default=36, help='batch_size per gpu')
parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
parser.add_argument('--deterministic', type=int, default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float, default=3e-4,
                    help='segmentation network learning rate')
parser.add_argument('--img_size', type=int,
                    default=192, help='input patch size of network input')
parser.add_argument('--seed', type=int,
                    default=1234, help='random seed')
parser.add_argument('--plant', type=str,
                    default='all', help='random seed')
# parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )
parser.add_argument(
    "--opts",
    help="Modify config options by adding 'KEY VALUE' pairs. ",
    default=None,
    nargs='+',
)
parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                    help='no: no cache, '
                         'full: cache all data, '
                         'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
parser.add_argument('--resume', help='resume from checkpoint')
parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
parser.add_argument('--use-checkpoint', action='store_true',
                    help="whether to use gradient checkpointing to save memory")
parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                    help='mixed precision opt level, if O0, no amp is used')
parser.add_argument('--tag', help='tag of experiment')
parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
parser.add_argument('--throughput', action='store_true', help='Test throughput only')
parser.add_argument('--gpu_id', type=str, default='0')
parser.add_argument('--num_workers', type=int, default=0)

args = parser.parse_args()
if args.dataset == "Synapse":
    args.root_path = os.path.join(args.root_path, "train_npz")
# config = get_config(args)


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    dataset_name = args.dataset
    dataset_config = {
        'Brats19': {
            'root_path': args.root_path,
            'list_dir': args.list_dir,
            'num_classes': 4,
        },
    }

    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.root_path = dataset_config[dataset_name]['root_path']
    args.list_dir = dataset_config[dataset_name]['list_dir']

    args.output_dir = os.path.join(args.output_dir, time.strftime("%m%d-%H%M"))
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    config_save_path = os.path.join(args.output_dir, 'train_config.yaml')
    save_args_to_config(args, config_save_path)

    net = MISSFormer(num_classes=args.num_classes).cuda(0)

    trainer = {'Brats19': trainer_brats19, }
    trainer[dataset_name](args, net)
