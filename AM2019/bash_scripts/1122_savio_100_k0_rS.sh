#!/bin/bash
# Job name:
#SBATCH --job-name=20_k0_rS_v100
#
# Account:
#SBATCH --account=fc_battery
#
# Partition:
#SBATCH --partition=savio3_gpu
#SBATCH --qos=v100_gpu3_normal
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:V100:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=juy16thu@berkeley.edu
#
# Wall clock limit (5 minutes here):
#SBATCH --time=72:00:00
#
## Command(s) to run:
python run.py --graph_size 100 --run_name k0_rel_svoV100 --no_progress_bar\
 --non_Euc --rank_k_approx 0 --rescale_dist --who YJ