save_dir="nE2024_data"
num_graphs=5000
seed=2024
# python make_rnd_dataset.py --graph_size 100 --num_graphs $num_graphs --seed $seed --save_dir $save_dir --force_triangle_iter 4
# python make_rnd_dataset.py --graph_size 100 --num_graphs $num_graphs --seed $seed --save_dir $save_dir --force_triangle_iter 4 --no_coords
python make_rnd_dataset.py --graph_size 100 --num_graphs $num_graphs --seed $seed --save_dir $save_dir --force_triangle_iter 4 --problem cvrp

# rename file "rnd_N100_I10000_S_seed2024_iter10_NoTrack.pkl" as "Val_rnd_N100_I10000_S_seed2024_iter10.pkl"
# mv "rnd_N100_I10000_S_seed2024_iter10_NoTrack.pkl" "Val_rnd_N100_I10000_S_seed2024_iter10.pkl"

# python make_rnd_dataset.py --graph_size 100 --num_graphs $num_graphs --seed $seed --save_dir $save_dir --mini_copy --force_triangle_iter 4 --problem cvrp