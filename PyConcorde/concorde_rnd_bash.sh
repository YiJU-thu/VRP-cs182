data_dir="/home/ecal_team/datasets/amazon_vrp/amazon_processed"
# python concorde_solve.py --type ATSP --data_dir $data_dir --data rnd_N20_I1000_S_seed1234_iter4.pkl \
#  --redo_failed --out_prefix CCD --clear 100 --save 100 --scale 1000 --big_M 50 --log CCD_rnd_20.log
python concorde_solve.py --type ATSP --data_dir $data_dir --data rnd_N50_I1000_S_seed1234_iter4.pkl\
 --redo_failed --out_prefix CCD --clear 10 --save 10 --scale 1000 --big_M 50 --log CCD_rnd_50.log
python concorde_solve.py --type ATSP --data_dir $data_dir --data rnd_N100_I1000_S_seed1234_iter4.pkl\
 --redo_failed --out_prefix CCD --clear 1 --save 1 --scale 1000 --big_M 50 --log CCD_rnd_100.log