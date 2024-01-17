graph_size=50
save_dir="Fu2021_dataset/atsp_${graph_size}"
num_graphs=10000
for seed in {0..9}
do
    python make_rnd_dataset.py --graph_size $graph_size --num_graphs $num_graphs --seed $seed \
    --save_dir $save_dir --force_triangle_iter 4
done