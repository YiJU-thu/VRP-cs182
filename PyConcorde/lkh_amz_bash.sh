data_dir="/home/ecal_team/datasets/amazon_vrp/amazon_processed"
tmp_dir="LKH_amz_NoTrack"
python concorde_solve.py --type ATSP --data_dir $data_dir --data amazon_eval_N20_I1000_seed1234_S.pkl --solver lkh\
 --redo_failed --out_prefix LKH --clear 100 --save 100 --scale 10 --log LKH_20.log --tmp_log_dir $tmp_dir
python concorde_solve.py --type ATSP --data_dir $data_dir --data amazon_eval_N50_I1000_seed1234_S.pkl --solver lkh\
 --redo_failed --out_prefix LKH --clear 100 --save 100 --scale 10 --log LKH_50.log --tmp_log_dir $tmp_dir
python concorde_solve.py --type ATSP --data_dir $data_dir --data amazon_eval_N100_I1000_seed1234_S.pkl --solver lkh\
 --out_prefix LKH --clear 10 --save 10 --scale 10 --log LKH_100.log --tmp_log_dir $tmp_dir

python concorde_solve.py --type ATSP --data_dir $data_dir --data amazon_eval_N20_I1000_seed1234_S.pkl --solver lkh\
 --redo_failed --out_prefix LKH_nS --clear 100 --save 100 --scale 10 --log LKH_20.log --skip_station --tmp_log_dir $tmp_dir
python concorde_solve.py --type ATSP --data_dir $data_dir --data amazon_eval_N50_I1000_seed1234_S.pkl --solver lkh\
 --redo_failed --out_prefix LKH_nS --clear 100 --save 100 --scale 10 --log LKH_50.log --skip_station --tmp_log_dir $tmp_dir
python concorde_solve.py --type ATSP --data_dir $data_dir --data amazon_eval_N100_I1000_seed1234_S.pkl --solver lkh\
 --out_prefix LKH_nS --clear 10 --save 10 --scale 10 --log LKH_100.log --skip_station --tmp_log_dir $tmp_dir
# python concorde_solve.py --type ATSP --data_dir $data_dir --data amazon_eval.pkl\
#  --redo_failed --out_prefix LKH --clear 1 --save 10 --scale 10 --big_M 200 --log LKH_eval.log
# python concorde_solve.py --type ATSP --data_dir $data_dir --data amazon_train.pkl\
#  --redo_failed --out_prefix LKH --clear 1 --save 10 --scale 10 --big_M 100 --log LKH_train.log