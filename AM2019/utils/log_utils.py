def log_values(cost, grad_norms, epoch, batch_id, step,
               log_likelihood, reinforce_loss, bl_loss, tb_logger, wandb_logger, opts):
    avg_cost = cost.mean().item()
    grad_norms, grad_norms_clipped = grad_norms

    # Log values to screen
    print('epoch: {}, train_batch_id: {}, avg_cost: {}'.format(epoch, batch_id, avg_cost))

    print('grad_norm: {}, clipped: {}'.format(grad_norms[0], grad_norms_clipped[0]))

    if opts.baseline == 'pomo': # log the variance of equivalent instances
        N1, N2 = opts.pomo_sample, opts.rot_sample
        c_reshaped = cost.view(-1, N1*N2)
        div_pct = ((c_reshaped.max(dim=1)[0] - c_reshaped.min(dim=1)[0]) / c_reshaped.mean(dim=1)).mean().item()


    # Log values to tensorboard
    if not opts.no_tensorboard:
        tb_logger.log_value('avg_cost', avg_cost, step)

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
    
    # Log values to wandb
    if not opts.no_wandb:
        wandb_logger.log({'avg_cost': avg_cost,
                          'actor_loss': reinforce_loss.item(),
                          'nll': -log_likelihood.mean().item(),
                          'grad_norm': grad_norms[0],
                          'grad_norm_clipped': grad_norms_clipped[0],
                          })

        if opts.baseline == 'critic':
            wandb_logger.log({'critic_loss': bl_loss.item(),
                              'critic_grad_norm': grad_norms[1],
                              'critic_grad_norm_clipped': grad_norms_clipped[1],
                              })
        
        if opts.baseline == 'pomo':
            wandb_logger.log({'pomo_div_pct': div_pct})
