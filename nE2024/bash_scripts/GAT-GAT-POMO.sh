#!/bin/bash
# Job name:
#SBATCH --job-name=gat_gat
#
# Account:
#SBATCH --account=fc_battery
#
# Partition:
#SBATCH --partition=savio4_gpu
#SBATCH --qos=a5k_gpu4_normal
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:A5000:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=juy16thu@berkeley.edu
#
# Wall clock limit (5 minutes here):
#SBATCH --time=72:00:00
#
## Command(s) to run:
# submitted: k=0, 5
optim="adamW"
val_ds="../dataset/nE2024_data/rnd_N100_I5000_S_seed2024_iter4_NoTrack.pkl"


python run.py --graph_size 100 --run_name "GAT-GAT-pomo32-b800" --no_progress_bar --non_Euc --who YJ\
 --batch_size 800 --batch_per_epoch 2000 --n_encode_layers 6 --n_edge_encode_layers 6 --encode_original_edge\
 --baseline pomo --pomo_sample 32 --optimizer $optim --weight_decay_model 1e-6 --val_dataset $val_ds

# python run.py --graph_size 100 --run_name "GAT-GAT-pomo32-K20" --no_progress_bar --non_Euc --who YJ\
#  --batch_size 640 --batch_per_epoch 2000 --n_encode_layers 6 --n_edge_encode_layers 6 --encode_original_edge\
#  --baseline pomo --pomo_sample 32 --optimizer $optim --weight_decay_model 1e-6 --val_dataset $val_ds\
#  --rank_k_approx 20 --mul_sigma_uv


# python run.py --graph_size 100 --run_name "GAT-GAT-adamW-L3" --no_progress_bar --non_Euc --who YJ\
#  --batch_size 256 --batch_per_epoch 2000 --n_encode_layers 3 --n_edge_encode_layers 3 --encode_original_edge\
#  --optimizer $optim --val_dataset ../dataset/nE2024_data/rnd_N100_I5000_S_seed2024_iter4_NoTrack.pkl

# for (( i=200; i>=20; i-=20 ))
# do
#     bs=$((32 * i))
#     python run.py --graph_size 100 --run_name "GAT-GAT-pomo32-b{$i}" --no_progress_bar --non_Euc --who YJ\
#      --batch_size $bs --batch_per_epoch 2000 --n_encode_layers 6 --n_edge_encode_layers 6 --encode_original_edge\
#      --baseline pomo --pomo_sample 32 --optimizer $optim --val_dataset ../dataset/nE2024_data/rnd_N100_I5000_S_seed2024_iter4_NoTrack.pkl
# done