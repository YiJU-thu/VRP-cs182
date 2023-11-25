#!/bin/bash
# Job name:
#SBATCH --job-name=20_k3_1080
#
# Account:
#SBATCH --account=fc_battery
#
# Partition:
#SBATCH --partition=savio2_1080ti
#SBATCH --qos=savio_normal
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=juy16thu@berkeley.edu
#
# Wall clock limit (5 minutes here):
#SBATCH --time=72:00:00
#
## Command(s) to run:
python run.py --graph_size 20 --run_name k3_rel_svo1080 --no_progress_bar\
 --non_Euc --rank_k_approx 3 --who YJ