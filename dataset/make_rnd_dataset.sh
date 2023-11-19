save_dir="/home/ecal_team/datasets/amazon_vrp/amazon_processed"
num_graphs=1000
seed=1234
python make_rnd_dataset.py --graph_size 20 --num_graphs $num_graphs --seed $seed --save_dir $save_dir --mini_copy
python make_rnd_dataset.py --graph_size 50 --num_graphs $num_graphs --seed $seed --save_dir $save_dir --mini_copy 
python make_rnd_dataset.py --graph_size 100 --num_graphs $num_graphs --seed $seed --save_dir $save_dir --mini_copy  