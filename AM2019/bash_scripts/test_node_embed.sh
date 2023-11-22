# python run.py --problem tsp --non_Euc --graph_size 20 --epoch_size 5120\
#  --rank_k_approx 3 --log_step 5 --run_name test_node_embed --no_wandb
python run.py --problem tsp --non_Euc --graph_size 20 --epoch_size 5120\
 --rank_k_approx 3 --rescale_dist --log_step 5 --run_name test_node_embed --no_wandb