# python run.py --problem tsp --non_Euc --graph_size 20 --epoch_size 5120\
#  --rank_k_approx 3 --log_step 5 --run_name test_node_embed --no_wandb
python run.py --problem tsp --non_Euc --graph_size 100 --val_size 512 --epoch_size 12800\
 --rank_k_approx 5 --rand_dist complex --log_step 5 --run_name test_node_embed --no_wandb