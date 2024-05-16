import os
import time
import argparse
import torch
from utils.functions import parse_softmax_temperature


def get_options(args=None):
    parser = argparse.ArgumentParser(
        description="Attention based model for solving the Travelling Salesman Problem with Reinforcement Learning")

    # [Data]
    parser.add_argument('--problem', default='tsp', help="The problem to solve, default 'tsp'")
    parser.add_argument('--non_Euc', action='store_true', help="Whether the problem is non-Euclidean. If the case, both coords and distance matrix will be provided")
    parser.add_argument('--no_coords', action='store_true', help="Whether to use coordinates in the model")
    parser.add_argument('--graph_size', type=int, default=20, help="The size of the problem graph")
    parser.add_argument('--batch_size', type=int, default=512, help='Number of instances per batch during training')
    parser.add_argument('--batch_per_epoch', type=int, default=2000, help='Number of batches per epoch during training')
    # parser.add_argument('--epoch_size', type=int, default=1280000, help='Number of instances per epoch during training. This will be overwritten by batch_size*batch_per_epoch')
    parser.add_argument('--val_size', type=int, default=10000,
                        help='Number of instances used for reporting validation performance')
    parser.add_argument('--val_dataset', type=str, default=None, help='Dataset file to use for validation')
    parser.add_argument('--rand_dist', type=str, default='standard', help='"standard" or "complex"') # FIXME: can be combined with data_distribution
    
    
    # [Model]
    parser.add_argument('--encoder', default='gat', help="Encoder name, 'gat' (default) or 'gcn'")
    parser.add_argument('--decoder', default='gat', help="Decoder name, 'gat' (default) or 'nAR'")

    # GAT encoder kwargs
    gat_init_encoder_kws = ["embedding_dim", "hidden_dim", "problem", "non_Euc", 
                       "rank_k_approx", "svd_original_edge", "mul_sigma_uv", "full_svd", "only_distance", "no_coords", "random_node_dim"]
    gat_encoder_kws = gat_init_encoder_kws + ["n_edge_encode_layers", "encode_original_edge", "rescale_dist", "n_encode_layers", 
                       "normalization", "n_heads", "checkpoint_encoder", "return_heatmap", "umat_embed_layers", "aug_graph_embed_layers"]

    parser.add_argument('--embedding_dim', type=int, default=128, help='Dimension of input embedding')
    parser.add_argument('--hidden_dim', type=int, default=128, help='Dimension of hidden layers in Enc/Dec')
    parser.add_argument('--n_encode_layers', type=int, default=3,
                        help='Number of layers in the encoder/critic network')
    parser.add_argument('--n_heads', type=int, default=8, help='Number of heads in attention layer')
    parser.add_argument('--normalization', default='batch', help="Normalization type, 'batch' (default) or 'instance'")
    parser.add_argument('--rank_k_approx', type=int, default=0, help='compute rank k-approx of dist matrix to argument node features')
    parser.add_argument('--n_edge_encode_layers', type=int, default=0, help='add edge matrix encodings to the first n attention layers')
    parser.add_argument('--encode_original_edge', action='store_true', help='if not, encode the relative distance matrix')
    parser.add_argument('--svd_original_edge', action='store_true', help='if not, do SVD on the relative distance matrix')
    parser.add_argument('--full_svd', action='store_true', help='if not, use randomized algorithm to perform faster SVD')
    parser.add_argument('--mul_sigma_uv', action='store_true', help='if True, add sqrt(sigma) u, sqrt(sigma) v to the node features')
    parser.add_argument('--only_distance', action='store_true', help='if True, do not use coordinates in the model') # compatible with rank_k_approx > 0 & svd_original_edge = True
    parser.add_argument('--return_heatmap', action='store_true', help='if True, return the heatmap instead of embeddings (Decoder use nAR)')
    parser.add_argument('--umat_embed_layers', type=int, default=3, help='number of MLP hidden layers for umat embedding')
    parser.add_argument('--aug_graph_embed_layers', type=int, default=3, help='number of MLP hidden layers for augmented graph embedding')
    parser.add_argument('--rescale_dist', action='store_true', help='if rand_dist is not standard, whether to rescale it to standard')
    parser.add_argument('--random_node_dim', type=int, default=0, help='randomly generate initial node features of this dimension in U(0,1)')
    parser.add_argument('--checkpoint_encoder', action='store_true',
                        help='Set to decrease memory usage by checkpointing encoder')

    
    # GCN encoder kwargs
    gcn_init_encoder_kws = gat_init_encoder_kws + ["edge_embedding_dim", "adj_mat_embedding_dim", "kNN"]
    gcn_encoder_kws = gcn_init_encoder_kws + ["encode_original_edge", "rescale_dist", "n_encode_layers", "normalization", 
                                              "checkpoint_encoder", "return_heatmap", "umat_embed_layers", "aug_graph_embed_layers", "gcn_aggregation"]
    parser.add_argument('--edge_embedding_dim', type=int, default=None, help='Dimension of edge embedding')
    parser.add_argument('--adj_mat_embedding_dim', type=int, default=None, help='Dimension of adjacency matrix embedding')
    parser.add_argument('--kNN', type=int, default=20, help='Number of nearest neighbors to consider in the adjacency matrix')
    parser.add_argument('--gcn_aggregation', default='sum', help="Aggregation type, 'mean' or 'sum' (default)")
    
    
    # GAT decoder kwargs 
    gat_decoder_kws = ["embedding_dim", "problem", "update_context_node", "tanh_clipping", "mask_inner", "mask_logits", "n_heads", "shrink_size"]
    nAR_decoder_kws = ["problem", "tanh_clipping", "mask_logits", "shrink_size"]

    parser.add_argument('--tanh_clipping', type=float, default=10.,
                        help='Clip the parameters to within +- this value using tanh. '
                             'Set to 0 to not perform any clipping.')
    parser.add_argument('--update_context_node', action='store_true', help='if True, use the context node instead of graph embedding for next step context node')
    parser.add_argument('--shrink_size', type=int, default=None,
                        help='Shrink the batch size if at least this many instances in the batch are finished'
                             ' to save memory (default None means no shrinking)')

    parser.add_argument('--pomo_sample', type=int, default=1, help='number of samples for pomo')
    parser.add_argument('--rot_sample', type=int, default=1, help='number of samples for Sym-NCO')
    parser.add_argument('--shpp', action='store_true', help='if True, train in SHPP mode: fix the first two steps')
    parser.add_argument('--shpp_skip', type=int, default=5, help='when training with SHPP, train original TSP every shpp_skip batches. 0 means inf')
    
    # parser.add_argument('--aug_graph_embed', action='store_true', help='this is due to a previous mistake. we should always set it as true, while this may ruin the previous trained models')


    # [Training]
    parser.add_argument('--learning_scheme', default='RL', help='Learning scheme: valid: RL, SL, USL')
    # SL only
    parser.add_argument('--tot_samples', type=int, default=None, help='Total number of samples for training')
    parser.add_argument('--augmentation', type=str, default='none', help='Data augmentation method. Options: none, flip, rotate, roll, or any combo. linked w/ "_"')
    parser.add_argument('--n_aug', type=int, default=1, help='Number of augmentations per sample.')
    parser.add_argument('--n_loaded_files', type=int, default=10, help='Number of files loaded at once')
    parser.add_argument('--start_file_idx', type=int, default=0, help='Index of the first file to load')
    parser.add_argument('--sl_debug', action='store_true', help='Debug mode for SL')

    parser.add_argument('--optimizer', default='adam', help="Optimizer to use, 'adam' (default) or 'adamW'")
    parser.add_argument('--lr_model', type=float, default=1e-4, help="Set the learning rate for the actor network")
    parser.add_argument('--lr_critic', type=float, default=1e-4, help="Set the learning rate for the critic network")
    parser.add_argument('--lr_decay', type=float, default=1.0, help='Learning rate decay per epoch')
    parser.add_argument('--weight_decay_model', type=float, default=0, help='Weight decay (L2 penalty) for the actor network')
    parser.add_argument('--weight_decay_critic', type=float, default=0, help='Weight decay (L2 penalty) for the critic network')
    parser.add_argument('--eval_only', action='store_true', help='Set this value to only evaluate model')
    parser.add_argument('--n_epochs', type=int, default=300, help='The number of epochs to train')
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
    parser.add_argument('--val_beam_size', type=int, default=100, help='Beam size to use in evaluation after each epoch')
    parser.add_argument('--data_distribution', type=str, default=None,
                        help='Data distribution to use during training, defaults and options depend on problem.')
    parser.add_argument('--best_val', type=float, default=None, help='Best validation score so far')

    # [Misc]
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
    # replace "none", "None", "NONE" with None
    for k, v in vars(opts).items():
        if v in ["none", "None", "NONE"]:
            setattr(opts, k, None)


    if opts.encoder == 'gat':
        opts.encoder_kwargs = {k: v for k, v in vars(opts).items() if k in gat_encoder_kws}
    elif opts.encoder == 'gcn':
        opts.encoder_kwargs = {k: v for k, v in vars(opts).items() if k in gcn_encoder_kws}
    
    if opts.decoder == 'gat':
        opts.decoder_kwargs = {k: v for k, v in vars(opts).items() if k in gat_decoder_kws}
        assert opts.return_heatmap == False, "heatmap is only used in nAR decoder"
    elif opts.decoder == 'nAR':
        opts.decoder_kwargs = {k: v for k, v in vars(opts).items() if k in nAR_decoder_kws}
        assert opts.return_heatmap == True, "heatmap is only used in nAR decoder"
    else:
        raise NotImplementedError


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
    # assert opts.epoch_size % opts.batch_size == 0, "Epoch size must be integer multiple of batch size!"
    
    opts.pomo_sample = 1 if opts.pomo_sample is None else opts.pomo_sample
    opts.rot_sample = 1 if opts.rot_sample is None else opts.rot_sample

    # adjust batch_size & epoch_size for POMO
    # so the input batch size is still opts.batch_size (true batch size)
    N1, N2 = opts.pomo_sample, opts.rot_sample
    opts.batch_size = opts.batch_size // (N1*N2)
    opts.epoch_size = opts.batch_size * opts.batch_per_epoch

    if opts.shpp:
        opts.force_steps = 2
    elif opts.pomo_sample > 1:
        opts.force_steps = 1
    else:
        opts.force_steps = 0
    opts.force_steps_batch = opts.force_steps
    
    
    if opts.learning_scheme == 'SL':
        curr_dir = os.path.dirname(os.path.realpath(__file__))
        assert opts.baseline is None
        
        # find the dataset
        if opts.sl_debug:
            dir = os.path.join(curr_dir, '../dataset/nE2024_data/test')
            instance_per_file = 5000
            fns = [f"rnd_N100_I5000_S_seed{i}_iter4_NoTrack.pkl" for i in range(2)]
            filenames = {
                os.path.join(dir, fns[i]): os.path.join(dir, "CCD_"+fns[i]) for i in range(2)
            }
        else:
            dir = os.path.join(curr_dir, '../dataset/nE2024_data/SL_100')
            instance_per_file = 10000
            fns = [f"rnd_N100_I10000_S_seed{i+100:03d}_iter4_NoTrack.pkl" for i in range(100)]
            filenames = {
                os.path.join(dir, f"SL_{i//10:03d}", fns[i]): os.path.join(dir, f"SL_{i//10:03d}", "LKH_"+fns[i]) for i in range(100)
            }
        fns = list(filenames.keys())
        
        if opts.tot_samples is not None:
            n_files = opts.tot_samples // instance_per_file
            if n_files > len(fns):
                raise ValueError("Not enough files for the given tot_samples")
            else:
                fns = fns[:n_files]
                filenames = {k: filenames[k] for k in fns}
        opts.sl_filenames = filenames




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
