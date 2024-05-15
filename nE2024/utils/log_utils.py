def log_values(cost, grad_norms, epoch, batch_id, step,
               log_likelihood, reinforce_loss, bl_loss, time_stats, tb_logger, wandb_logger, opts):
    avg_cost = cost.mean().item() if cost is not None else None
    grad_norms, grad_norms_clipped = grad_norms

    # Log values to screen
    print('epoch: {}, train_batch_id: {}, avg_cost: {}'.format(epoch, batch_id, avg_cost))

    print('grad_norm: {}, clipped: {}'.format(grad_norms[0], grad_norms_clipped[0]))

    if opts.baseline == 'pomo': # log the variance of equivalent instances
        N1, N2 = opts.pomo_sample, opts.rot_sample
        c_reshaped = cost.view(-1, N1*N2)
        c_max, c_min, c_mean = c_reshaped.max(dim=1)[0], c_reshaped.min(dim=1)[0], c_reshaped.mean(dim=1)
        div_pct = ((c_max - c_min) / c_mean).mean().item()


    # Log values to tensorboard
    if not opts.no_tensorboard:
        if avg_cost is not None:
            tb_logger.log_value('avg_cost', avg_cost, step)
        if reinforce_loss is not None:
            tb_logger.log_value('actor_loss', reinforce_loss.item(), step)
        
        tb_logger.log_value('nll', -log_likelihood.mean().item(), step)

        tb_logger.log_value('grad_norm', grad_norms[0], step)
        tb_logger.log_value('grad_norm_clipped', grad_norms_clipped[0], step)

        if opts.baseline == 'critic':
            tb_logger.log_value('critic_loss', bl_loss.item(), step)
            tb_logger.log_value('critic_grad_norm', grad_norms[1], step)
            tb_logger.log_value('critic_grad_norm_clipped', grad_norms_clipped[1], step)
        
        if opts.baseline == 'pomo':
            tb_logger.log_value('pomo_div_pct', div_pct, step)
            tb_logger.log_value('pomo_avg_cost', c_min.mean().item(), step)
    
    # Log values to wandb
    log_info = {
        # 'avg_cost': avg_cost,
        # 'actor_loss': reinforce_loss.item(),
        'nll': -log_likelihood.mean().item(),
        'grad_norm': grad_norms[0],
        'grad_norm_clipped': grad_norms_clipped[0],
    }
    if avg_cost is not None:
        log_info['avg_cost'] = avg_cost
    if reinforce_loss is not None:
        log_info['actor_loss'] = reinforce_loss.item()
        
    if opts.baseline == 'critic':
        log_info.update({
            'critic_loss': bl_loss.item(),
            'critic_grad_norm': grad_norms[1],
            'critic_grad_norm_clipped': grad_norms_clipped[1],
        })
    if opts.baseline == 'pomo':
        log_info.update({
            'pomo_div_pct': div_pct,
            'pomo_avg_cost': c_min.mean().item(),
        })
    log_info.update(time_stats)
    
    
    if not opts.no_wandb:
        wandb_logger.log(log_info)  # "step" in wandb is the step of logging
