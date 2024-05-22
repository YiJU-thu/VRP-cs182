data_dir="../dataset/nE2024_data/test"


# larger dataset
# sizes=(200 500 1000)
sizes=(200)
# Iterate over each size
for size in "${sizes[@]}"
do
    python concorde_solve.py --type ATSP --data_dir $data_dir --out_dir $data_dir --data rnd_N${size}_I100_S_seed0_iter4_NoTrack.pkl \
        --redo_failed --out_prefix CCD --clear 1 --save 1 --scale 1000 --big_M 50 --log CCD_rnd_${size}.log --I 100 --time_bound 1800
done


for ((i=0; i<=1; i++))
do
    python concorde_solve.py --dask_parallel --type ATSP --data_dir $data_dir --out_dir $data_dir --data rnd_N100_I5000_S_seed${i}_iter4_NoTrack.pkl \
        --redo_failed --out_prefix CCD --clear 10 --save 10 --scale 1000 --big_M 50 --log CCD_rnd_100.log --n_workers 10
done

for ((i=0; i<=1; i++))
do
    python concorde_solve.py --dask_parallel --type ATSP --data_dir $data_dir --out_dir $data_dir --data rnd_N100_I5000_C_seed${i}_iter4_NoTrack.pkl \
        --redo_failed --out_prefix CCD --clear 10 --save 10 --scale 1000 --big_M 50 --log CCD_rnd_100.log --recover_graph --n_workers 16
done