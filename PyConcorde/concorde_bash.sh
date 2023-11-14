data_dir="/home/ecal_team/datasets/amazon_vrp/amazon_processed"
python concorde_solve.py --type ATSP --data_dir $data_dir --data amazon_eval_N20_I1000_seed1234_S.pkl\
 --redo_failed --out_prefix CCD --clear 100 --save 100 --scale 10 --big_M 200 --log CCD_20.log
python concorde_solve.py --type ATSP --data_dir $data_dir --data amazon_eval_N50_I1000_seed1234_S.pkl\
 --redo_failed --out_prefix CCD --clear 100 --save 100 --scale 10 --big_M 100 --log CCD_50.log
python concorde_solve.py --type ATSP --data_dir $data_dir --data amazon_eval_N100_I1000_seed1234_S.pkl\
 --redo_failed --out_prefix CCD --clear 10 --save 10 --scale 10 --big_M 100 --log CCD_100.log

## debug command
# python concorde_solve.py --type ATSP --I 100 --data_dir $data_dir --data amazon_eval_N20_I1000_seed1234_S_mini100.pkl --redo_failed --out_prefix TEST --clear 20 --save 20 --scale 10 --big_M 200 