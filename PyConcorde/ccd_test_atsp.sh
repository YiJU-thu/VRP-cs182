data_dir="../dataset/nE2024_data/test"
for ((i=0; i<=1; i++))
do
    python concorde_solve.py --dask_parallel --type ATSP --data_dir $data_dir --out_dir $data_dir --data rnd_N100_I5000_S_seed${i}_iter4_NoTrack.pkl \
        --redo_failed --out_prefix CCD --clear 10 --save 10 --scale 1000 --big_M 50 --log CCD_rnd_100.log
done