data_dir="../dataset/nE2024_data/test"

# MatNet style ATSP
python concorde_solve.py --dask_parallel --type ATSP --data_dir $data_dir --out_dir $data_dir --data rnd_N100_I10000_NC_seed0_iter4_NoTrack.pkl  --solver lkh\
    --redo_failed --out_prefix LKH --clear 10 --save 10 --scale 1 --log LKH_rnd_100.log --n_workers 10

for ((i=0; i<=1; i++))
do
    python concorde_solve.py --dask_parallel --type ATSP --data_dir $data_dir --out_dir $data_dir --data rnd_N100_I5000_S_seed${i}_iter4_NoTrack.pkl --solver lkh\
        --redo_failed --out_prefix LKH --clear 10 --save 10 --scale 1000 --log LKH_rnd_100.log --n_worker 10
done

for ((i=0; i<=1; i++))
do
    python concorde_solve.py --dask_parallel --type ATSP --data_dir $data_dir --out_dir $data_dir --data rnd_N100_I5000_C_seed${i}_iter4_NoTrack.pkl --solver lkh\
        --redo_failed --out_prefix LKH --clear 10 --save 10 --scale 1000 --log LKH_rnd_100.log --recover_graph --n_workers 16
done