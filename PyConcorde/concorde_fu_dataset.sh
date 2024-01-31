graph_size=50
data_dir="../dataset/Fu2021_dataset/atsp_${graph_size}"
for seed in {0..9}
do
    python concorde_solve.py --type ATSP --data_dir $data_dir --data "rnd_N${graph_size}_I10000_S_seed${seed}_iter4_NoTrack.pkl" \
    --redo_failed --out_prefix CCD --out_dir $data_dir --clear 100 --save 100 --scale 1000 --big_M 50 --log CCD_rnd_${graph_size}.log \
    --ignore_NoTrack
done