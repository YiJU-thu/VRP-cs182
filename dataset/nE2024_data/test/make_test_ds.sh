# run this under dataset directory
save_dir="nE2024_data/test"

# Simple-dist non-Euc TSP test dataset
num_graphs=5000
for ((i=0; i<=1; i++))
do
    python make_rnd_dataset.py --graph_size 100 --num_graphs $num_graphs --seed $i --save_dir $save_dir --force_triangle_iter 4
done

# Simple-dist MatNet style TSP test dataset
num_graphs=10000
i=0
python make_rnd_dataset.py --graph_size 100 --num_graphs $num_graphs --seed $i --save_dir $save_dir --force_triangle_iter 4 --no_coords

# Complex-dist non-Euc TSP test dataset
num_graphs=5000
for ((i=0; i<=1; i++))
do
    python make_rnd_dataset.py --graph_size 100 --num_graphs $num_graphs --rescale --seed $i --save_dir $save_dir --force_triangle_iter 4
done

# Simple-dist non-Euc CVRP test dataset
num_graphs=1000
i=0
python make_rnd_dataset.py --problem cvrp --graph_size 100 --num_graphs $num_graphs --seed $i --save_dir $save_dir --force_triangle_iter 4

# Euclidean TSP test dataset
num_graphs=10000
python make_rnd_dataset.py --Euc --graph_size 100 --num_graphs $num_graphs --seed 0 --save_dir $save_dir