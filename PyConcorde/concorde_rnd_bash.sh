data_dir="/home/ecal_team/datasets/amazon_vrp/amazon_processed"
# python concorde_solve.py --type ATSP --data_dir $data_dir --data rnd_N20_I1000_seed1234.pkl --time_bound 1\
#  --redo_failed --out_prefix CCD --clear 10 --save 1 --scale 1000 --big_M 100 --log CCD_rnd_20.log
python concorde_solve.py --type ATSP --data_dir $data_dir --data rnd_N50_I1000_seed1234.pkl\
 --redo_failed --out_prefix CCD --clear 10 --save 1 --scale 1000 --big_M 100 --log CCD_rnd_50.log
# python concorde_solve.py --type ATSP --data_dir $data_dir --data rnd_N100_I1000_seed1234.pkl --time_bound 120\
#  --redo_failed --out_prefix CCD --clear 1 --save 1 --scale 1000 --big_M 60 --log CCD_rnd_100.log