# run this under dataset directory
save_dir="nE2024_data/test"
num_graphs=5000
for ((i=0; i<=1; i++))
do
    python make_rnd_dataset.py --graph_size 100 --num_graphs $num_graphs --seed $i --save_dir $save_dir --force_triangle_iter 4
done