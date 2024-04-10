#!/bin/bash
#SBATCH --partition=savio2
#SBATCH --account=fc_battery
#SBATCH --ntasks=24
#SBATCH --time=72:00:00
module load python
# export SCHED=$(hostname)
# dask-scheduler&
# sleep 10
# # Start one worker per SLURM 'task' (i.e., core)
# srun dask-worker tcp://${SCHED}:8786 &   # might need ${SCHED}.berkeley.edu
# echo ${SCHED}
# sleep 20
data_dir="../dataset/nE2024_data/SL_100"
for ((i=101; i<=120; i++))
do
python concorde_solve.py --dask_parallel --type ATSP --data_dir $data_dir --out_dir $data_dir --data rnd_N100_I5000_S_seed${i}_iter4_NoTrack.pkl \
--redo_failed --out_prefix CCD --clear 10 --save 10 --scale 1000 --big_M 50 --log CCD_rnd_100.log --n_workers 10
done