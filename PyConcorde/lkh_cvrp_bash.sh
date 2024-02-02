data_dir="/home/ecal_team/datasets/amazon_vrp/amazon_processed"
tmp_dir="LKH_cvrp_NoTrack"
python concorde_solve.py --problem cvrp --type ATSP --capacity 30 --data_dir $data_dir --data CVRP_rnd_N20_I1000_S_seed1234_iter4.pkl --solver lkh\
 --redo_failed --out_prefix LKH --clear 10 --save 10 --scale 1000 --log LKH_cvrp_20.log --tmp_log_dir "${tmp_dir}_20"