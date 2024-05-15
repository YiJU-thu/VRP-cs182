import os
import time
from tqdm import tqdm
import torch
import math

from torch.utils.data import DataLoader
from torch.nn import DataParallel

from nets.decoder_gat import set_decode_type
from utils.log_utils import log_values
from utils import move_to, gpu_memory_usage

import time
from loguru import logger

def get_inner_model(model):
    return model.module if isinstance(model, DataParallel) else model

def validate(model, dataset, opts):
    # Validate
    print('Validating with greedy...')
    cost = rollout(model, dataset, opts, force_steps=0, decode_type="greedy")
    avg_cost = cost.mean()
    print('Validation overall avg_cost: {} +- {}'.format(
        avg_cost, torch.std(cost) / math.sqrt(len(cost))))

    # beam search
    width = opts.val_beam_size
    if width is None or width <= 1:
        return avg_cost, None
    print(f'First 100 - greedy: {cost[:100].mean()}')
    print('Validating with beam search...')
    cost = rollout(model, dataset, opts, force_steps=0, decode_type="bs")
    avg_cost_bs = cost.mean()
    print('Validation [beam search {}] overall avg_cost: {} +- {}'.format(
        width, avg_cost_bs, torch.std(cost) / math.sqrt(len(cost))))
    return avg_cost, avg_cost_bs


def rollout(model, dataset, opts, force_steps=0, decode_type="greedy"):
    # Put in greedy evaluation mode!
    assert decode_type in ["greedy", "bs"]
    set_decode_type(model, decode_type)
    model.eval()

    def eval_model_bat(bat):
        with torch.no_grad():
            cost, _ = model(move_to(bat, opts.device), force_steps=force_steps)
        return cost.data.cpu()

    def eval_model_bat_bs(bat, width):
        with torch.no_grad():
            seq, cost = model.beam_search(move_to(bat, opts.device), beam_size=width, compress_mask=False, max_calc_batch_size=opts.eval_batch_size)
        return torch.tensor(cost).data.cpu()


    val = []
    batch_size = opts.eval_batch_size if decode_type == "greedy" else 100 # to save time, we only evaluate the first 100 samples
    for batch in tqdm(DataLoader(dataset, batch_size=batch_size), disable=opts.no_progress_bar):
        if not opts.rescale_dist:
            if 'scale_factors' in batch.keys():
                batch['scale_factors'] = None
            elif 'data' in batch.keys():
                batch['data']['scale_factors'] = None
            else:
                raise ValueError("No scale_factors in batch")
        if decode_type == "greedy":
            val.append(eval_model_bat(batch))
        elif decode_type == "bs":
            val.append(eval_model_bat_bs(batch, opts.val_beam_size))
            break # only evaluate the first batch


    return torch.cat(val, 0)


def clip_grad_norms(param_groups, max_norm=math.inf):
    """
    Clips the norms for all param groups to max_norm and returns gradient norms before clipping
    :param optimizer:
    :param max_norm:
    :param gradient_norms_log:
    :return: grad_norms, clipped_grad_norms: list with (clipped) gradient norms per group
    """
    grad_norms = [
        torch.nn.utils.clip_grad_norm_(
            group['params'],
            max_norm if max_norm > 0 else math.inf,  # Inf so no clipping but still call to calc
            norm_type=2
        )
        for group in param_groups
    ]
    grad_norms_clipped = [min(g_norm, max_norm) for g_norm in grad_norms] if max_norm > 0 else grad_norms
    return grad_norms, grad_norms_clipped


def train_epoch(model, dataloader, optimizer, baseline, lr_scheduler, epoch, val_dataset, problem, tb_logger, wandb_logger, opts):
    print("Start train epoch {}, lr={} for run {}".format(epoch, optimizer.param_groups[0]['lr'], opts.run_name))
    step = epoch * (opts.epoch_size // opts.batch_size)
    start_time = time.time()

    if not opts.no_tensorboard:
        tb_logger.log_value('learnrate_pg0', optimizer.param_groups[0]['lr'], step)

    # # Generate new training data for each epoch (FIXME: skipped to avoid memory crash)
    # training_dataset = baseline.wrap_dataset(problem.make_dataset(
    #     size=opts.graph_size, num_samples=opts.epoch_size, non_Euc=opts.non_Euc, 
    #     rand_dist=opts.rand_dist, rescale=opts.rescale_dist, distribution=opts.data_distribution))
    # training_dataloader = DataLoader(training_dataset, batch_size=opts.batch_size, num_workers=1)

    # Put model in train mode!
    model.train()
    set_decode_type(model, "sampling")

    # Yi: generate an epoch_size dataset may crash the memory, we skip the dataloader
    #   and generate a batch_size dataset for each batch
    n_batches = opts.epoch_size // opts.batch_size
    for batch_id in range(n_batches):
        
        gpu_memory_usage(msg=f"batch_id: {batch_id} - START")

        if opts.shpp and opts.shpp_skip != 0:
            opts.force_steps_batch = opts.force_steps * (batch_id%opts.shpp_skip != 1)   # do not force steps for every opts.shpp_skip batches
            # so the initial place holder gets a chance to be trained to find good second step (only apply for shpp mode)
            # choose ==1 so that logged batch val is based on SHPP (while epoch metric is TSP)

        t0 = time.perf_counter()
        
        with torch.device("cpu"): # make sure the dataset is on CPU
            if dataloader is not None:
                batch, sol = next(dataloader)
            else:
                batch_dataset = baseline.wrap_dataset(problem.make_dataset(
                    size=opts.graph_size, num_samples=opts.batch_size, non_Euc=opts.non_Euc, 
                    rand_dist=opts.rand_dist, rescale=opts.rescale_dist, distribution=opts.data_distribution))
                    
                for batch in tqdm(DataLoader(batch_dataset, batch_size=len(batch_dataset)), disable=opts.no_progress_bar):
                    break # only need the first batch
                sol = None

        model.update_time_count(data_gen=time.perf_counter()-t0)    # record data generation time

        gpu_memory_usage(msg=f"batch_id: {batch_id} - DATA LOADED")

    # for batch_id, batch in enumerate(tqdm(training_dataloader, disable=opts.no_progress_bar)):
        if not opts.rescale_dist:
            if 'scale_factors' in batch.keys():
                batch['scale_factors'] = None
            elif 'data' in batch.keys():
                batch['data']['scale_factors'] = None
            else:
                raise ValueError("No scale_factors in batch")


        train_batch(
            model,
            optimizer,
            baseline,
            epoch,
            batch_id,
            step,
            batch,
            sol,
            tb_logger,
            wandb_logger,
            opts
        )

        step += 1

    epoch_duration = time.time() - start_time
    print("Finished epoch {}, took {} s".format(epoch, time.strftime('%H:%M:%S', time.gmtime(epoch_duration))))

    def save_model(save_dir, fn, epoch):
        print('Saving model and state...')
        torch.save(
            {
                'model': get_inner_model(model).state_dict(),
                'optimizer': optimizer.state_dict(),
                'rng_state': torch.get_rng_state(),
                'cuda_rng_state': torch.cuda.get_rng_state_all(),
                'baseline': baseline.state_dict(),
                'epoch': epoch,
            },
            os.path.join(save_dir, fn)
        )
        logger.success(f"{fn} saved")
        
    def clear_model_states(save_dir, fn):
        fn_p = os.path.join(save_dir, fn)
        if os.path.exists(fn_p):
            saved = torch.load(fn_p)
            torch.save({
                'model': saved['model'],
                'epoch': saved['epoch'],
                },
                fn_p
            )
            logger.success(f"{fn} cleared")


    if (opts.checkpoint_epochs != 0 and epoch % opts.checkpoint_epochs == 0) or epoch == opts.n_epochs - 1:
        print('Saving model and state...')
        fn_save = 'epoch-{}.pt'.format(epoch)
        save_model(opts.save_dir, fn_save, epoch)
        fn_clear = 'epoch-{}.pt'.format(epoch - opts.checkpoint_epochs)
        clear_model_states(opts.save_dir, fn_clear)
        
    avg_reward, avg_reward_bs = validate(model, val_dataset, opts)
    opts.best_val = opts.best_val if opts.best_val is not None else avg_reward+1
    if avg_reward < opts.best_val:
        opts.best_val = avg_reward
        fn_best = 'best.pt'
        save_model(opts.save_dir, fn_best, epoch)
        logger.success(f"{fn_best} [{epoch}] saved")

    if not opts.no_tensorboard:
        tb_logger.log_value('val_avg_reward', avg_reward, step)
    if not opts.no_wandb:
        info = {'val_avg_reward': avg_reward, 'epoch': epoch}
        if avg_reward_bs is not None:
            info['val_avg_reward_bs'] = avg_reward_bs
        wandb_logger.log(info)

    baseline.epoch_callback(model, epoch)

    # lr_scheduler should be called at end of epoch
    lr_scheduler.step()


def train_batch(
        model,
        optimizer,
        baseline,
        epoch,
        batch_id,
        step,
        batch,
        sol,
        tb_logger,
        wandb_logger,
        opts
):
    
    if opts.learning_scheme == 'RL':
        x, bl_val = baseline.unwrap_batch(batch)
        x = move_to(x, opts.device)
        bl_val = move_to(bl_val, opts.device) if bl_val is not None else None

        gpu_memory_usage(msg=f"batch_id: {batch_id} - DATA MOVED TO DEVICE")

        # Evaluate model, get costs and log probabilities
        cost, log_likelihood = model(x, force_steps=opts.force_steps_batch)

        gpu_memory_usage(msg=f"batch_id: {batch_id} - MODEL EVALUATED")

        t0 = time.perf_counter()
        # Evaluate baseline, get baseline loss if any (only for critic)
        bl_val, bl_loss = baseline.eval(x, cost) if bl_val is None else (bl_val, 0)
        model.update_time_count(baseline_eval=time.perf_counter()-t0)    # record baseline evaluation time

        # Calculate loss
        reinforce_loss = ((cost - bl_val) * log_likelihood).mean()
        loss = reinforce_loss + bl_loss
    
    elif opts.learning_scheme == 'SL':
        x, ref_pi = batch, sol
        x, ref_pi = move_to(x, opts.device), move_to(ref_pi, opts.device)
        ref_pi = ref_pi.long()  # convert to int64
        cost, log_likelihood = model(x, ref_pi=ref_pi)
        loss = -log_likelihood.mean()  # maximize the likelihood of the reference sequence (optimal tours)
        cost, reinforce_loss, bl_loss = None, None, None    # no meaning for SL

    elif opts.learning_scheme == 'USL':
        raise NotImplementedError("Unsupervised Learning is not implemented yet")
    else:
        raise ValueError("Unknown learning scheme, please choose from ['RL', 'SL', 'USL']")

    t0 = time.perf_counter()
    # Perform backward pass and optimization step
    optimizer.zero_grad()
    loss.backward()
    # Clip gradient norms and get (clipped) gradient norms for logging
    grad_norms = clip_grad_norms(optimizer.param_groups, opts.max_grad_norm)
    optimizer.step()
    model.update_time_count(model_update=time.perf_counter()-t0)    # record model update time
    
    gpu_memory_usage(msg=f"batch_id: {batch_id} - BACKWARD PASS")

    # Logging
    if step % int(opts.log_step) == 0:
        time_stats = model.time_stats
        log_values(cost, grad_norms, epoch, batch_id, step,
                   log_likelihood, reinforce_loss, bl_loss, time_stats, tb_logger, wandb_logger, opts)
