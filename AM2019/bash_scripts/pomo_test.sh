# python run.py --problem tsp --graph_size 20 --val_size 512 --epoch_size 5120 --baseline pomo --pomo_sample 8 --log_step 5 --run_name test_pomo --no_wandb
# python run.py --problem tsp --graph_size 20 --val_size 512 --epoch_size 5120 --baseline pomo --rot_sample 8 --log_step 5 --run_name test_rot --no_wandb
# python run.py --problem tsp --graph_size 20 --val_size 512 --epoch_size 5120 --baseline pomo --pomo_sample 4 --rot_sample 2 --log_step 5 --run_name test_pomo_rot --no_wandb
python run.py --problem tsp --graph_size 20 --non_Euc --val_size 512 --epoch_size 5120 --baseline pomo --pomo_sample 4 --rot_sample 2 --log_step 5 --run_name test_pomo_rot --no_wandb
# python run.py --problem tsp --graph_size 20 --val_size 512 --epoch_size 5120 --log_step 5 --run_name test_no_pomo --no_wandb