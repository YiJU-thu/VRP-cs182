# run this under dataset directory
save_dir="nE2024_data/SL_100"

# Simple-dist Euc TSP test dataset
num_graphs=100000
python make_rnd_dataset.py --Euc --graph_size 100 --num_graphs $num_graphs --seed 101 --save_dir $save_dir

# Simple-dist non-Euc TSP test dataset
num_graphs=5000
for ((i=101; i<=120; i++))
do
    python make_rnd_dataset.py --graph_size 100 --num_graphs $num_graphs --seed $i --save_dir $save_dir --force_triangle_iter 4
done

