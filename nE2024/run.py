#!/usr/bin/env python

import os
import json
import pprint as pp

import torch
import torch.optim as optim
from tensorboard_logger import Logger as TbLogger
# from torchsummary import summary

# from nets.critic_network import CriticNetwork
from options import get_options
from train import train_epoch, validate, get_inner_model
from reinforce_baselines import NoBaseline, ExponentialBaseline, CriticBaseline, RolloutBaseline, WarmupBaseline, PomoBaseline
# from nets.attention_model import AttentionModel
# from nets.pointer_network import PointerNetwork, CriticNetworkLSTM

from nets.encoder_decoder import VRPModel

from utils import torch_load_cpu, load_problem
from utils.vrp_dataset import VRPDataset, VRPLargeDataset

from loguru import logger

import wandb
API_KEY = os.environ.get("WANDB_API_KEY")
# you should set WANDB_API_KEY in your environment before running this script
# using the command: export WANDB_API_KEY=<your_api_key_here>

@logger.catch
def run(opts):
    project = opts.project
    # nE = non-Euclidean, rS=rescale_dist
    logger.add(f"logs/{project}.log", rotation="10 MB")

    # Pretty print the run args
    pp.pprint(vars(opts))
    logger.info(f"Project: {project}")
    logger.info(vars(opts))

    # Set the random seed
    torch.manual_seed(opts.seed)

    # Optionally configure tensorboard
    tb_logger = None
    if not opts.no_tensorboard:
        tb_logger = TbLogger(os.path.join(opts.log_dir, "{}_{}".format(opts.problem, opts.graph_size), opts.run_name))

    os.makedirs(opts.save_dir)
    # Save arguments so exact configuration can always be found
    with open(os.path.join(opts.save_dir, "args.json"), 'w') as f:
        json.dump(vars(opts), f, indent=True)


    # init wandb to track this run
    if not opts.no_wandb:
        wandb.login(key=API_KEY)
        wandb_logger = wandb.init(
            entity=opts.wandb_entity,   # a username, or a team name
            project=project,     
            name=opts.run_name, 
            config=vars(opts))
    else:
        wandb_logger = None


    # Set the device
    opts.device = torch.device("cuda:0" if opts.use_cuda else "cpu")

    # Figure out what's the problem
    problem = load_problem(opts.problem)

    # Load data from load_path
    load_data = {}
    assert opts.load_path is None or opts.resume is None, "Only one of load path and resume can be given"
    load_path = opts.load_path if opts.load_path is not None else opts.resume
    if load_path is not None:
        print('  [*] Loading data from {}'.format(load_path))
        load_data = torch_load_cpu(load_path)

    
    opts.encoder_kwargs['problem'] = problem
    opts.decoder_kwargs['problem'] = problem

    model = VRPModel(
        encoder_name=opts.encoder,
        decoder_name=opts.decoder,
        encoder_kws=opts.encoder_kwargs,
        decoder_kws=opts.decoder_kwargs
    ).to(opts.device)

    if opts.use_cuda and torch.cuda.device_count() > 1:
        logger.info(f"Using {torch.cuda.device_count()} GPUs!")
        # model = torch.nn.parallel.DistributedDataParallel(model)
        model = torch.nn.DataParallel(model)

    # Overwrite model parameters by parameters to load
    model_ = get_inner_model(model)
    model_.load_state_dict({**model_.state_dict(), **load_data.get('model', {})})

    # print model summary
    logger.info(model)
    logger.info(f"TOTAL PARAMS: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    # save model summary to txt file
    with open(os.path.join(opts.save_dir, "model_summary.txt"), "w") as f:
        f.write(str(model))
        f.write(f"\n\nPARAMS: {[p.numel() for p in model.parameters() if p.requires_grad]}")
        f.write(f"\n\nTOTAL PARAMS: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    logger.info(f"Model summary saved")



    # Initialize baseline
    if opts.baseline == 'exponential':
        baseline = ExponentialBaseline(opts.exp_beta)
    # elif opts.baseline == 'critic' or opts.baseline == 'critic_lstm':
    #     assert problem.NAME == 'tsp', "Critic only supported for TSP"
    #     baseline = CriticBaseline(
    #         (
    #             CriticNetworkLSTM(
    #                 2,
    #                 opts.embedding_dim,
    #                 opts.hidden_dim,
    #                 opts.n_encode_layers,
    #                 opts.tanh_clipping
    #             )
    #             if opts.baseline == 'critic_lstm'
    #             else
    #             CriticNetwork(
    #                 2,
    #                 opts.embedding_dim,
    #                 opts.hidden_dim,
    #                 opts.n_encode_layers,
    #                 opts.normalization
    #             )
    #         ).to(opts.device)
    #     )
    elif opts.baseline == 'rollout':
        baseline = RolloutBaseline(model, problem, opts)
    elif opts.baseline == 'pomo':
        baseline = PomoBaseline(model, problem, opts)
    else:
        assert opts.baseline is None, "Unknown baseline: {}".format(opts.baseline)
        baseline = NoBaseline()

    if opts.bl_warmup_epochs > 0:
        baseline = WarmupBaseline(baseline, opts.bl_warmup_epochs, warmup_exp_beta=opts.exp_beta)

    # Load baseline from data, make sure script is called with same type of baseline
    if 'baseline' in load_data:
        baseline.load_state_dict(load_data['baseline'])

    # Initialize optimizer
    optimizers = {"adam": optim.Adam, "adamW": optim.AdamW}
    optimizer = optimizers[opts.optimizer](
        [{'params': model.parameters(), 'lr': opts.lr_model, 'weight_decay': opts.weight_decay_model}]
        + (
            [{'params': baseline.get_learnable_parameters(), 'lr': opts.lr_critic, 'weight_decay': opts.weight_decay_critic}]
            if len(baseline.get_learnable_parameters()) > 0
            else []
        )
    )

    # Load optimizer state
    if 'optimizer' in load_data:
        optimizer.load_state_dict(load_data['optimizer'])
        for state in optimizer.state.values():
            for k, v in state.items():
                # if isinstance(v, torch.Tensor):
                if torch.is_tensor(v):
                    state[k] = v.to(opts.device)

    # Initialize learning rate scheduler, decay by lr_decay once per epoch!
    lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: opts.lr_decay ** epoch)

    # Start the actual training loop
    # FIXME: this dataset may be too large to fit in memory
    with torch.device("cpu"): # make sure the dataset is on CPU
        val_dataset = problem.make_dataset(
            size=opts.graph_size, num_samples=opts.val_size, filename=opts.val_dataset, normalize_loaded=False, # if load from file, do not (repeatedly) normalize
            non_Euc=opts.non_Euc, rand_dist=opts.rand_dist, rescale=opts.rescale_dist, distribution=opts.data_distribution, 
            no_coords=opts.no_coords, keep_rel=opts.keep_rel, force_triangle_iter=4)

    if opts.resume:
        epoch_resume = load_data.get('epoch')
        if epoch_resume is None:    # in old versions, epoch was not saved
            epoch_resume =  int(os.path.splitext(os.path.split(opts.resume)[-1])[0].split("-")[1])

        torch.set_rng_state(load_data['rng_state'])
        if opts.use_cuda:
            torch.cuda.set_rng_state_all(load_data['cuda_rng_state'])
        # Set the random states
        # Dumping of state was done before epoch callback, so do that now (model is loaded)
        baseline.epoch_callback(model, epoch_resume)
        print("Resuming after {}".format(epoch_resume))
        opts.epoch_start = epoch_resume + 1

    if opts.eval_only:
        validate(model, val_dataset, opts)
    else:
        if opts.learning_scheme == "SL":
            with torch.device("cpu"):
                dataloader = VRPLargeDataset(filenames=opts.sl_filenames, batch_size=opts.batch_size, n_loaded_files=opts.n_loaded_files, start_file_idx=opts.start_file_idx, 
                                            shuffle=True, augmentation=opts.augmentation, n_aug=opts.n_aug)
            dataloader = iter(dataloader)
        else:
            dataloader = None
        for epoch in range(opts.epoch_start, opts.epoch_start + opts.n_epochs):
            train_epoch(
                model,
                dataloader,
                optimizer,
                baseline,
                lr_scheduler,
                epoch,
                val_dataset,
                problem,
                tb_logger,
                wandb_logger,
                opts
            )

    if not opts.no_wandb:
        wandb.finish()

if __name__ == "__main__":
    run(get_options())