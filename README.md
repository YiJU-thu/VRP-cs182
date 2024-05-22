# Benchmarking Deep-learning-based Vehicle Routing Problem (VRP) Solvers on Non-Euclidean Instances


### training
- dependencies: when you are at `nE2024/`, you can run
```
python env_checker.py
```

# suppose you are at /AM2019
python run.py --graph_size 20 --non_Euc --run_name 'nE_tsp20_test' --wandb_entity <your wandb username> (or --no_wandb)
```
- for model options, please refer to `AM2019/options.py`

### evaluation
- under `nE2024/pretrained/neurips`, we provide our trained model. (we remove the optimizer states to save space, so you cannot resume training from this checkpoint, but only use for inference)\
for evaluation, please use methods provided in `nE2024/nE_vrp_eval.py`

For example:
you can run:
```
python nE_vrp_eval.py --log_fn none --I 100 --model nE_tsp100_GAT-GAT-pomo32/epoch-199.pt --ds rnd_S --decode_strategy sgbs --width 2 --gamma 3 --max_calc_batch_size 10000
```