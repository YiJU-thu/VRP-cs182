data_dir="../dataset/nE2024_data/SL_100"
i=101
python concorde_solve.py --dask_parallel --type EUC_2D --data_dir $data_dir --out_dir $data_dir --data rnd_N100_I100000_EUC_S_seed${i}_iter0_NoTrack.pkl \
--redo_failed --out_prefix CCD --clear 20 --save 20 --scale 1000 --log CCD_rnd_100.log --n_workers 10