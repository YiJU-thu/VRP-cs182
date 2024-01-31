# big_M starts from 100, then go to 200, 500, etc..., as small big_M can solve faster, but may return invalid solution
data_dir="/home/ecal_team/datasets/amazon_vrp/amazon_processed"
python concorde_solve.py --type ATSP --data_dir $data_dir --data amazon_eval_N20_I1000_seed1234_S.pkl\
 --redo_failed --out_prefix CCD_nS --clear 100 --save 100 --scale 10 --big_M 1000 --log CCD_20.log --skip_station
# python concorde_solve.py --type ATSP --data_dir $data_dir --data amazon_eval_N50_I1000_seed1234_S.pkl\
#  --redo_failed --out_prefix CCD_nS --clear 100 --save 100 --scale 10 --big_M 500 --log CCD_50.log --skip_station
python concorde_solve.py --type ATSP --data_dir $data_dir --data amazon_eval_N100_I1000_seed1234_S.pkl\
 --redo_failed --out_prefix CCD_nS --clear 1 --save 1 --scale 10 --big_M 300 --log CCD_100.log --skip_station
# python concorde_solve.py --type ATSP --data_dir $data_dir --data amazon_eval.pkl\
#  --redo_failed --out_prefix CCD --clear 1 --save 10 --scale 10 --big_M 200 --log CCD_eval.log
# python concorde_solve.py --type ATSP --data_dir $data_dir --data amazon_train.pkl\
#  --redo_failed --out_prefix CCD --clear 1 --save 10 --scale 10 --big_M 100 --log CCD_train.log

## debug command
# python concorde_solve.py --type ATSP --I 100 --data_dir $data_dir --data amazon_eval_N20_I1000_seed1234_S_mini100.pkl --redo_failed --out_prefix TEST --clear 20 --save 20 --scale 10 --big_M 200 