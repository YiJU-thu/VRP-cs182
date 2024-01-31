data_dir="/home/ecal_team/datasets/amazon_vrp/amazon_processed"
tmp_dir="LKH_rnd_NoTrack"
python concorde_solve.py --type ATSP --data_dir $data_dir --data rnd_N20_I1000_S_seed1234_iter4.pkl --solver lkh\
 --redo_failed --out_prefix LKH --clear 100 --save 100 --scale 1000 --log LKH_rnd_20.log --tmp_log_dir $tmp_dir
python concorde_solve.py --type ATSP --data_dir $data_dir --data rnd_N50_I1000_S_seed1234_iter4.pkl --solver lkh\
 --redo_failed --out_prefix LKH --clear 10 --save 10 --scale 1000 --log LKH_rnd_50.log --tmp_log_dir $tmp_dir
python concorde_solve.py --type ATSP --data_dir $data_dir --data rnd_N100_I1000_S_seed1234_iter4.pkl --solver lkh\
 --redo_failed --out_prefix LKH --clear 1 --save 1 --scale 1000 --log LKH_rnd_100.log --tmp_log_dir $tmp_dir

python concorde_solve.py --type ATSP --data_dir $data_dir --data rnd_N20_I1000_C_seed1234_iter4.pkl --solver lkh\
 --redo_failed --out_prefix LKH --clear 100 --save 100 --scale 1000 --log LKH_rnd_20.log --recover_graph --tmp_log_dir $tmp_dir
python concorde_solve.py --type ATSP --data_dir $data_dir --data rnd_N50_I1000_C_seed1234_iter4.pkl --solver lkh\
 --redo_failed --out_prefix LKH --clear 10 --save 10 --scale 1000 --log LKH_rnd_50.log --recover_graph --tmp_log_dir $tmp_dir
python concorde_solve.py --type ATSP --data_dir $data_dir --data rnd_N100_I1000_C_seed1234_iter4.pkl --solver lkh\
 --redo_failed --out_prefix LKH --clear 1 --save 1 --scale 1000 --log LKH_rnd_100.log --recover_graph --tmp_log_dir $tmp_dir