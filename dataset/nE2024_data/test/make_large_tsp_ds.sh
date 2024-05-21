# run this under dataset directory
save_dir="nE2024_data/test"
num_graphs=100
sizes=(200 500 1000)
# Iterate over each size
for size in "${sizes[@]}"
do
    python make_rnd_dataset.py --graph_size $size --num_graphs $num_graphs --seed 0 --save_dir $save_dir --force_triangle_iter 4
done