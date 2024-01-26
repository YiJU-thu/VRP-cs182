#!/usr/bin/env python

import os
import json
import pprint as pp
import time

import torch
import torch.optim as optim
from tensorboard_logger import Logger as TbLogger

from utils.process import main
from utils.google_tsp_reader import BatchIterator

from loguru import logger

import wandb
API_KEY = os.environ.get("WANDB_API_KEY")
# you should set WANDB_API_KEY in your environment before running this script
# using the command: export WANDB_API_KEY=<your_api_key_here>

import argparse


@logger.catch
def run(config):
    data_generator = BatchIterator(
        filepaths=config.train_filepaths,
        num_nodes = config.num_nodes, num_neighbors = config.num_neighbors,
        batch_size=config.batch_size, 
        shuffled=False, augmentation=False, aug_prob=config.aug_prob,
    )

    
    
    
    if not config.no_wandb:
        wandb.login(key=API_KEY)
        wandb_logger = wandb.init(
            entity=config.wandb_entity,   # a username, or a team name
            project=config.project,     
            name=config.run_name, 
            config=vars(config))
    else:
        wandb_logger = None

    main(config, iter(data_generator), wandb_logger)

    if wandb_logger is not None:
        wandb_logger.finish()


def get_options(args=None):
    parser = argparse.ArgumentParser(
        description='Att-GCRN+MCTS model for solving TSP')

    # Data
    parser.add_argument('--data_dir', type=str, default=None, help='root data directory for all datasets')
    parser.add_argument('--train_filepath', type=str, default=None, help='Path to training data file')
    parser.add_argument('--train_sol_filepath', type=str, default=None, help='...')
    parser.add_argument('--train_file_num', type=int, default=1, help='...')
    parser.add_argument('--val_filepath', type=str, default=None, help='Path to validation data file')
    parser.add_argument('--val_sol_filepath', type=str, default=None, help='Path to validation data file')
    parser.add_argument('--val_file_num', type=int, default=1, help='...')
    # parser.add_argument('--test_filepath', type=str, default='...', help='Path to test data file')
    # parser.add_argument('--test_sol_filepath', type=str, default=None, help='Path to test data file')
    # parser.add_argument('--test_file_num', type=int, default=1, help='...')

    # Model
    parser.add_argument('--netname', type=str, default='att-gcn', help='Name of the model. att-gcn or am')
    
    parser.add_argument('--num_nodes', type=int, default=20, help='Number of nodes in TSP')
    parser.add_argument('--num_neighbors', type=int, default=-1, help='Number of neighbors in TSP. -1 for fully connected')
    parser.add_argument('--node_dim', type=int, default=2, help='Dimension of node features')
    parser.add_argument('--voc_nodes_in', type=int, default=2, help='Number of node features')
    parser.add_argument('--voc_nodes_out', type=int, default=2, help='Number of node features')
    parser.add_argument('--voc_edges_in', type=int, default=3, help='Number of edge features')
    parser.add_argument('--voc_edges_out', type=int, default=2, help='Number of edge features')
    parser.add_argument('--beam_size', type=int, default=1280, help='Beam size for beam search')
    parser.add_argument('--hidden_dim', type=int, default=300, help='Hidden dimension')
    parser.add_argument('--num_layers', type=int, default=30, help='Number of GCN layers')
    parser.add_argument('--mlp_layers', type=int, default=3, help='Number of MLP layers')
    parser.add_argument('--aggregation', type=str, default='mean', help='Aggregation function for node features')
    parser.add_argument('--max_epochs', type=int, default=1500, help='Maximum number of epochs')
    parser.add_argument('--val_every', type=int, default=1, help='Validation frequency')
    parser.add_argument('--test_every', type=int, default=100, help='Test frequency')
    parser.add_argument('--batch_size', type=int, default=20, help='Batch size')
    parser.add_argument('--batches_per_epoch', type=int, default=500, help='Number of batches per epoch')
    parser.add_argument('--accumulation_steps', type=int, default=1, help='Number of batches for gradient accumulation')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--decay_rate', type=float, default=1.01, help='Learning rate decay')
    parser.add_argument('--num_neg', type=int, default=10, help='Number of negative samples')
    parser.add_argument('--gamma', type=float, default=1, help='Gamma for Focal Loss')
    parser.add_argument('--loss_type', type=str, default='FL', help='Loss type')
    parser.add_argument('--aug_prob', type=float, default=1.00, help='Augmentation probability')


    # TODO: Attention Edge encoding parameters

    parser.add_argument('--patience', type=int, default=1, help='Patience for early stopping')
    parser.add_argument('--lr_scale', type=float, default=1., help='Learning rate scale for pretrained model')
    parser.add_argument('--var_neighbor', type=int, default=5, help='Number of neighbors for variance')
    parser.add_argument('--random_neighbor', action='store_true', help='Use random neighbors for variance')


    parser.add_argument('--pretrained', action='store_true', help='Use pretrained model')
    parser.add_argument('--pretrained_path', type=str, default=None, help='Path to pretrained model')   # FIXME: use 'resume' instead



    # Misc
    parser.add_argument('--log_step', type=int, default=50, help='Log info every log_step steps')
    parser.add_argument('--log_dir', default='logs', help='Directory to write TensorBoard information to')
    parser.add_argument('--run_name', default='run', help='Name to identify the run')
    parser.add_argument('--output_dir', default='outputs', help='Directory to write output models to')
    parser.add_argument('--epoch_start', type=int, default=0,
                        help='Start at epoch # (relevant for learning rate decay)')
    parser.add_argument('--checkpoint_epochs', type=int, default=1,
                        help='Save checkpoint every n epochs (default 1), 0 to save no checkpoints')
    parser.add_argument('--load_path', help='Path to load model parameters and optimizer state from')
    parser.add_argument('--resume', help='Resume from previous checkpoint file')
    parser.add_argument('--no_tensorboard', action='store_true', help='Disable logging TensorBoard files')
    parser.add_argument('--no_wandb', action='store_true', help='Disable logging to wandb')
    parser.add_argument('--wandb_entity', default='ecal_ml4opt', help='Wandb entity (team space to submit logs)')
    parser.add_argument('--who', default=None, help='have a signiture for the person submit the run. e.g., YJ')
    parser.add_argument('--no_progress_bar', action='store_true', help='Disable progress bar')

    config = parser.parse_args(args)

    non_Euc = True; problem = "tsp"; rand_dist = 'standard'
    config.project = "{}{}_{}{}".format("nE_"*(non_Euc), problem, config.num_nodes, "_rS"*(rand_dist != 'standard'))
    
    config.run_name = "{}_{}".format(config.run_name, time.strftime("%Y%m%dT%H%M%S"))
    config.save_dir = os.path.join(config.output_dir, config.project, config.run_name)
    config.log_dir = os.path.join(config.log_dir, config.project, config.run_name)

    # filepaths
    # (1) original: one file end with .txt (only train_filepath)
    # (2) suggested: a pair of .pkl files, one for instance, one for solution (train_filepath & train_sol_filepath)
    # (3) when the dataset contains multiple files (use * to represent the index)

    def get_filepaths(filepath, sol_filepath, file_num, data_dir):
        filepath = os.path.join(data_dir, filepath)
        if sol_filepath is not None:
            sol_filepath = os.path.join(data_dir, sol_filepath)

        if "*" in filepath:
            filepaths = [filepath.replace("*", str(i)) for i in range(file_num)]
            if sol_filepath is None:
                return filepaths
            sol_filepaths = [sol_filepath.replace("*", str(i)) for i in range(file_num)]
            return [(filepaths[i], sol_filepaths[i]) for i in range(file_num)]
        if sol_filepath is None:
            return filepath
        return (filepath, sol_filepath)
    
    if config.data_dir is None:
        config.data_dir = f'../../dataset/Fu2021_dataset/atsp_{config.num_nodes}'
    if config.train_filepath is None:
        config.train_filepath = f'rnd_N{config.num_nodes}_I10000_S_seed*_iter4_NoTrack.pkl'
        config.train_sol_filepath = f'CCD_rnd_N{config.num_nodes}_I10000_S_seed*_iter4.pkl'
        config.train_file_num = 9
    if config.val_filepath is None:
        config.val_filepath = f'rnd_N{config.num_nodes}_I10000_S_seed9_iter4_NoTrack.pkl'
        config.val_sol_filepath = f'CCD_rnd_N{config.num_nodes}_I10000_S_seed9_iter4.pkl'
        config.val_file_num = 1


    curr_dir = os.path.dirname(os.path.realpath(__file__))
    config.data_dir = os.path.join(curr_dir, config.data_dir)

    config.train_filepaths = get_filepaths(config.train_filepath, config.train_sol_filepath, config.train_file_num, config.data_dir)
    config.val_filepaths = get_filepaths(config.val_filepath, config.val_sol_filepath, config.val_file_num, config.data_dir)
    # config.test_filepaths = get_filepaths(config.test_filepath, config.test_sol_filepath, config.test_file_num, config.data_dir)

    return config



if __name__ == "__main__":
    run(get_options())

