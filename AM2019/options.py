import os
import time
import argparse
import torch
from utils.functions import parse_softmax_temperature


def get_options(args=None):
    parser = argparse.ArgumentParser(
        description="Attention based model for solving the Travelling Salesman Problem with Reinforcement Learning")

    # Data
    parser.add_argument('--problem', default='tsp', help="The problem to solve, default 'tsp'")
    parser.add_argument('--non_Euc', action='store_true', help="Whether the problem is non-Euclidean. If the case, both coords and distance matrix will be provided")
    parser.add_argument('--graph_size', type=int, default=20, help="The size of the problem graph")
    parser.add_argument('--batch_size', type=int, default=512, help='Number of instances per batch during training')
    parser.add_argument('--epoch_size', type=int, default=1280000, help='Number of instances per epoch during training')
    parser.add_argument('--val_size', type=int, default=10000,
                        help='Number of instances used for reporting validation performance')
    parser.add_argument('--val_dataset', type=str, default=None, help='Dataset file to use for validation')

    # Model
    parser.add_argument('--model', default='attention', help="Model, 'attention' (default) or 'pointer'")
    parser.add_argument('--embedding_dim', type=int, default=128, help='Dimension of input embedding')
    parser.add_argument('--hidden_dim', type=int, default=128, help='Dimension of hidden layers in Enc/Dec')
    parser.add_argument('--n_encode_layers', type=int, default=3,
                        help='Number of layers in the encoder/critic network')
    parser.add_argument('--tanh_clipping', type=float, default=10.,
                        help='Clip the parameters to within +- this value using tanh. '
                             'Set to 0 to not perform any clipping.')
    parser.add_argument('--normalization', default='batch', help="Normalization type, 'batch' (default) or 'instance'")
    parser.add_argument('--rank_k_approx', type=int, default=0, help='compute rank k-approx of dist matrix to argument node features')
    parser.add_argument('--n_edge_encode_layers', type=int, default=0, help='add edge matrix encodings to the first n attention layers')
    parser.add_argument('--encode_original_edge', action='store_true', help='if not, encode the relative distance matrix')
    parser.add_argument('--svd_original_edge', action='store_true', help='if not, do SVD on the relative distance matrix')
    parser.add_argument('--full_svd', action='store_true', help='if not, use randomized algorithm to perform faster SVD')
    parser.add_argument('--mul_sigma_uv', action='store_true', help='if True, add sqrt(sigma) u, sqrt(sigma) v to the node features')
    parser.add_argument('--only_distance', action='store_true', help='if True, do not use coordinates in the model') # compatible with rank_k_approx > 0 & svd_original_edge = True
    parser.add_argument('--rand_dist', type=str, default='standard', help='"standard" or "complex"') # FIXME: can be combined with data_distribution
    parser.add_argument('--rescale_dist', action='store_true', help='if rand_dist is not standard, whether to rescale it to standard')
    parser.add_argument('--pomo_sample', type=int, default=None, help='number of samples for pomo')
    parser.add_argument('--rot_sample', type=int, default=None, help='number of samples for Sym-NCO')

    # Training
    parser.add_argument('--lr_model', type=float, default=1e-4, help="Set the learning rate for the actor network")
    parser.add_argument('--lr_critic', type=float, default=1e-4, help="Set the learning rate for the critic network")
    parser.add_argument('--lr_decay', type=float, default=1.0, help='Learning rate decay per epoch')
    parser.add_argument('--weight_decay_model', type=float, default=0, help='Weight decay (L2 penalty) for the actor network')
    parser.add_argument('--weight_decay_critic', type=float, default=0, help='Weight decay (L2 penalty) for the critic network')
    parser.add_argument('--eval_only', action='store_true', help='Set this value to only evaluate model')
    parser.add_argument('--n_epochs', type=int, default=100, help='The number of epochs to train')
    parser.add_argument('--seed', type=int, default=1234, help='Random seed to use')
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                        help='Maximum L2 norm for gradient clipping, default 1.0 (0 to disable clipping)')
    parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA')
    parser.add_argument('--exp_beta', type=float, default=0.8,
                        help='Exponential moving average baseline decay (default 0.8)')
    parser.add_argument('--baseline', default='rollout',    # change the default baseline to 'rollout'
                        help="Baseline to use: 'rollout', 'pomo', 'critic' or 'exponential', None (no basleine). Defaults to rollout.")
    parser.add_argument('--bl_alpha', type=float, default=0.05,
                        help='Significance in the t-test for updating rollout baseline')
    parser.add_argument('--bl_warmup_epochs', type=int, default=None,
                        help='Number of epochs to warmup the baseline, default None means 1 for rollout (exponential '
                             'used for warmup phase), 0 otherwise. Can only be used with rollout baseline.')
    parser.add_argument('--eval_batch_size', type=int, default=1024,
                        help="Batch size to use during (baseline) evaluation")
    parser.add_argument('--checkpoint_encoder', action='store_true',
                        help='Set to decrease memory usage by checkpointing encoder')
    parser.add_argument('--shrink_size', type=int, default=None,
                        help='Shrink the batch size if at least this many instances in the batch are finished'
                             ' to save memory (default None means no shrinking)')
    parser.add_argument('--data_distribution', type=str, default=None,
                        help='Data distribution to use during training, defaults and options depend on problem.')

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



    opts = parser.parse_args(args)

    if opts.who is not None:
        opts.run_name = opts.run_name + '_' + opts.who


    opts.use_cuda = torch.cuda.is_available() and not opts.no_cuda
    opts.run_name = "{}_{}".format(opts.run_name, time.strftime("%Y%m%dT%H%M%S"))
    
    project = "{}{}_{}{}".format("nE_"*(opts.non_Euc), opts.problem, opts.graph_size, "_rS"*(opts.rand_dist != 'standard'))
    opts.project = project
    opts.save_dir = os.path.join(
        opts.output_dir,
        project,
        opts.run_name
    )
    if opts.bl_warmup_epochs is None:
        opts.bl_warmup_epochs = 1 if opts.baseline == 'rollout' else 0
    assert (opts.bl_warmup_epochs == 0) or (opts.baseline == 'rollout')
    assert opts.epoch_size % opts.batch_size == 0, "Epoch size must be integer multiple of batch size!"
    return opts


def get_eval_options(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs='+', help="Filename of the dataset(s) to evaluate")
    # parser.add_argument("-o", action='store_true', help="Set true to overwrite")
    # parser.add_argument("-f", default=None, help="Name of the results file to write")
    parser.add_argument('--val_size', type=int, default=10000,
                        help='Number of instances used for reporting validation performance')
    parser.add_argument('--offset', type=int, default=0,
                        help='Offset where to start in dataset (default 0)')
    parser.add_argument('--eval_batch_size', type=int, default=1024,
                        help="Batch size to use during (baseline) evaluation")
    # parser.add_argument('--decode_type', type=str, default='greedy',
    #                     help='Decode type, greedy or sampling')
    parser.add_argument('--width', type=int, nargs='+',
                        help='Sizes of beam to use for beam search (or number of samples for sampling), '
                             '0 to disable (default), -1 for infinite')
    parser.add_argument('--decode_strategy', default='greedy', type=str,
                        help='Beam search (bs), Sampling (sample) or Greedy (greedy)')
    parser.add_argument('--softmax_temperature', type=parse_softmax_temperature, default=1,
                        help="Softmax temperature (sampling or bs)")
    parser.add_argument('--model', type=str)
    parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA')
    parser.add_argument('--no_progress_bar', action='store_true', help='Disable progress bar')
    parser.add_argument('--compress_mask', action='store_true', help='Compress mask into long')
    parser.add_argument('--max_calc_batch_size', type=int, default=10000, help='Size for subbatches')
    parser.add_argument('--results_dir', default='results', help="Name of results directory")
    parser.add_argument('--multiprocessing', action='store_true',
                        help='Use multiprocessing to parallelize over multiple GPUs')

    opts = parser.parse_args(args)
    opts.f = None
    assert opts.f is None or (len(opts.datasets) == 1 and len(opts.width) <= 1), \
        "Cannot specify result filename with more than one dataset or more than one width"

    opts.widths = opts.width if opts.width is not None else [0]

    return opts
