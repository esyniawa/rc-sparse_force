import argparse
import os.path
import torch
from typing import Optional


def get_dataset_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_t', type=int, default=110,
                        help='Number of time steps')
    parser.add_argument('--num_init_thetas', type=int, default=500,
                        help='Number of initial joint angles')
    parser.add_argument('--num_goals', type=int, default=500,
                        help='Number of target positions')
    parser.add_argument('--wait_steps', type=int, default=10,
                        help='Wait steps after trajectory')
    parser.add_argument('--movement_duration', type=float, default=5.0,
                        help='Duration of movement')
    parser.add_argument('--train_split', type=float, default=0.8,
                        help='Train/eval split')
    parser.add_argument('--save_dir', type=str, default='arm_data',
                        help='Directory to save dataset')

    args = parser.parse_args()
    return args, parser


def get_training_args():
    """Define and parse command line arguments with default values."""
    parser = argparse.ArgumentParser(description='Train SpaRCe for Planar Arm Control')

    # Model parameters
    parser.add_argument('--dim_res', type=int, default=1000,
                        help='Reservoir dimension')
    parser.add_argument('--perc_n', type=float, default=50.0,
                        help='Percentile threshold')
    parser.add_argument('--prop_rec', type=float, default=0.01,
                        help='Proportion of recurrent connections')
    parser.add_argument('--spectral_radius', type=float, default=0.97,
                        help='Spectral radius')
    parser.add_argument('--ff_scale', type=float, default=0.5,
                        help='Scale of feedforward connections')
    parser.add_argument('--alpha', type=float, default=0.17,
                        help='Alpha parameter for leaky integrator')

    # Training parameters
    parser.add_argument('--sim_id', type=int, default=0,
                        help='Simulation id')
    parser.add_argument('--data_set_dir', type=str, default='arm_data',)
    parser.add_argument('--num_train_episodes', type=int, default=100,)
    parser.add_argument('--num_eval_episodes', type=int, default=10)
    parser.add_argument('--num_epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--device', type=torch.device, default='cpu',
                        help='Device (cuda or cpu)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')

    args = parser.parse_args()
    return args, parser


def save_args(args: argparse,
              save_name: str,
              additional_args: Optional[dict] = None,):

    if additional_args is not None:
        args.__dict__.update(additional_args)

    save_folder = os.path.split(save_name)[0]
    if save_folder != '':
        os.makedirs(save_folder, exist_ok=True)

    with open(save_name, 'w') as f:
        for name, value in vars(args).items():
            f.write(f"{name}: {value}\n")

    print(f'Arguments saved to {save_name}')


def import_args(save_name: str):
    """
    Import arguments from a file. All values are stored as strings. TODO: Add type conversion, probably by using the parser itself
    """
    with open(save_name, 'r') as f:
        args = argparse.Namespace()
        for line in f:
            name, value = line.strip().split(':')
            setattr(args, name, value)
    return args


if __name__ == "__main__":
    args, _ = get_training_args()
    save_args(args, 'args.txt')
    imported_args = import_args('args.txt')
    print(imported_args.num_epochs)