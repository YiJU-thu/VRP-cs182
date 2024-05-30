import numpy as np
import os, sys
import argparse
import pickle
from contextlib import redirect_stdout
from io import StringIO
import copy

from eval import _eval_dataset
from options import get_eval_options, get_options
from utils import load_model
import torch
from problems.tsp.tsp_utils import reorder_nodes, tour_len_euc_2d, load_data, load_concorde_ref


def get_res(model, data, shpp=False, no_progress_bar=True, no_gpu=True):
    
    n = data.shape[1]   # graph_size
    eval_batch_size = 1000 if n <= 200 else 100
    
    cmd = ["--eval_batch_size", str(eval_batch_size)] 
    if shpp:
        cmd.append("--shpp")
    if no_progress_bar:
        cmd += ["--no_progress_bar"]
    opts = get_eval_options(cmd)

    device = torch.device("cpu") if no_gpu else torch.device("cuda")    # FIXME: no parallelize over multiple GPUs
    width = 0
    softmax_temperature = 1

    model, _ = load_model(model)
    dataset = model.problem.make_dataset(data=data)
    res = _eval_dataset(model, dataset, width, softmax_temperature, opts, device)
    costs, tours, durations = zip(*res)
    
    costs = np.array(costs)
    tours = np.array(tours)
    durations = np.array(durations)

    return {"obj": costs, "tours": tours, "time": durations}


def refine_shpp(model, data, n_splits=2, idx_0=0, model_init=None, res0=None, mute=True):
    if model_init is None:
        model_init = model
    
    I, n = data.shape[:2]
    idx_0 = idx_0 % n
    
    # solve TSP with the init model to get an initial solution to be refined (or use the given res0)
    res0 = get_res(model_init, data) if res0 is None else res0  
    
    if not mute:
        print("Before refinement: {:.3f}+-{:.3f}".format(res0["costs"].mean(), res0["costs"].std()))

    tours0 = res0['tours']
    # equivalent to: shift (to left) idx_0 columns, and append the first column to the end
    tours0_expand = np.concatenate([tours0[:,idx_0:], tours0[:,:idx_0+1]], axis=1)
    nodes_tour0_expand = reorder_nodes(data, tours0_expand) # reorder the data accordingly


    len_split = np.ceil(data.shape[1] / n_splits).astype(int)   # each split has (approx) len_split nodes

    nodes_splits = [None] * n_splits
    idx_splits = [None] * n_splits
    start = 0

    for i in range(n_splits):   # FIXME: may improve later
        l = len_split + 1   # max(len_split + 1 + np.random.randint(-3,3), 1) # add some randomness to the length of each split
        end = min(start + l, nodes_tour0_expand.shape[1])
        if i == n_splits - 1:
            end = nodes_tour0_expand.shape[1]
        coords = nodes_tour0_expand[:, start:end, :]    # (I, (approx) len_split, 2)
        # map it into [0,1] x [0,1]
        xy_max = coords.max(axis=(1,2), keepdims=True)
        xy_min = coords.min(axis=(1,2), keepdims=True)
        coords = (coords - xy_min) / (xy_max - xy_min)
        # print("***", coords.max(axis=(1,2)).min(), coords.min(axis=(1,2)).max())
        nodes_splits[i] = coords
        idx_splits[i] = tours0_expand[:, start:end]
        start += l-1
    
    # solve SHPP for each split
    sol_splits = [get_res(model, nodes_splits[i], shpp=True) for i in range(n_splits)]
    durations = np.vstack([sol_splits[i]["time"] for i in range(n_splits)]).sum(axis=0)
    assert durations.shape == (I,)  # FIXME: the duration calculation seems wrong (miss a factor of n_split?)


    res_splits = [sol_splits[i]["tours"] for i in range(n_splits)]
    # in SHPP solution, nodes are (n, 1, ...), now, reorder them as (1, ..., n) so that they can be concatenated
    shpp_splits = [np.roll(res_splits[i], -1, axis=1) for i in range(n_splits)]
    tour_to_concat = [None] * n_splits
    for i in range(n_splits):
        s = shpp_splits[i]
        tour_to_concat[i] = idx_splits[i][range(s.shape[0]), s.T].T[:,:-1]
    tours1 = np.concatenate(tour_to_concat, axis=1)
    assert tours1.shape == tours0.shape, "tours1.shape={}, tours0.shape={}".format(tours1.shape, tours0.shape)
    obj1 = tour_len_euc_2d(data, tours1)

    res1 = {"obj": obj1, "tours": tours1, "time": res0["time"] + durations}
    
    if not mute:
        print("After refinement: {:.3f}+-{:.3f}".format(obj1.mean(), obj1.std()))
    
    return res1, res0


def eval_shpp_refine(model, data, n_splits=None, model_init=None,
                     eps=1e-3, iter_cmp=5, early_stop_pct=0.02, max_iter=100):
    
    model_init = model if model_init is None else model_init
    I, n = data.shape[:2]
    
    if n_splits is None:
        try:
            n_shpp = int(model.split("pretrained/tsp_")[1].split("/")[0])   # get the SHPP (training) graph size        
            n_splits = max(2, np.ceil(n / n_shpp)).astype(int)
        except:
            n_splits = 2
    print("n_splits:", n_splits)

    res0 = get_res(model_init, data)
    print("Before refinement: {:.3f}+-{:.3f}".format(res0["obj"].mean(), res0["obj"].std()))

    best = copy.deepcopy(res0)  # record the best solution (score & tour) so far
    score_hist = np.zeros((max_iter+1, I))    # record the best scores up to each iteration
    score_hist[0,:] = best["obj"]

    for i in range(max_iter):
        
        # Create a StringIO object to capture the output
        output_buffer = StringIO()
        with redirect_stdout(output_buffer):    
            res1, _ = refine_shpp(model, data, n_splits=n_splits, idx_0=i, res0=best)
        
        print(i, "After refinement: {:.3f}+-{:.3f}".format(res1["obj"].mean(), res1["obj"].std()))
        
        # updating best
        better = res1["obj"] < best["obj"] - eps
        best["obj"][better] = res1["obj"][better]
        best["tours"][better,:] = res1["tours"][better,:]
        best["time"] = res1["time"]
        # print("**", best["time"].mean())
        
        pct_improved_1 = better.mean()
        pct_improved_3 = (best["obj"]<res0["obj"]-eps).mean()

        if i >= iter_cmp:
            # percentage of instance that has improved in the last 10 rounds
            pct_improved_2 = (best["obj"] < score_hist[iter_cmp-1,:]-eps).mean()
        else:
            pct_improved_2 = pct_improved_3

        print(">", "Better: {:.3f}+-{:.3f} | {:.2%} (1) | {:.2%} ({}) | {:.2%} (infty)".format(
            best["obj"].mean(), best["obj"].std(), 
            pct_improved_1, pct_improved_2, iter_cmp, pct_improved_3))
        
        score_hist = np.roll(score_hist, 1, axis=0)   # the newer scores are on the upper rows
        score_hist[0,:] = best["obj"]

        if pct_improved_2 < early_stop_pct and i >= iter_cmp:
            print("Early stop at round", i)
            score_hist = score_hist[:i+2,:]
            break

    return best, score_hist


def get_save_fn(args):
    # FIXME: this anming is not comprehensive
    
    # tsp_20/epoch-99_shpp.pt -> 20-99_shpp
    m = args.model[4:-3].replace("/epoch", "")
    m0 = args.model_init[4:-3].replace("/epoch", "") if args.model_init is not None else m
    fn = f"d_{args.n}_s{args.seed}-M_{m}-M0_{m0}.pkl"
    return fn


if __name__ == "__main__":
    
    # sample:
    # python eval_shpp_refine.py --model tsp_20/epoch-99.pt --model_init tsp_20/epoch-99.pt --n 20 --I 1000 --seed 1234 --n_splits 2 --eps 1e-3 --iter_cmp 5 --early_stop_pct 0.02 --max_iter 100
    
    parser = argparse.ArgumentParser()
    
    # evaluation data
    parser.add_argument("--n", type=int, default=20, help="evaluate graph size")
    parser.add_argument("--I", type=int, default=None, help="number of instances to evaluate")
    parser.add_argument("--seed", type=int, default=1234, help="seed of the test data")
    
    # evaluation (refinement) models
    parser.add_argument("--model", type=str, required=True, help="model to solve SHPP")
    parser.add_argument("--model_init", type=str, default=None, help="initial model to generate solution before refinement")

    # refinement parameters
    parser.add_argument("--n_splits", type=int, default=None, help="number of splits")
    parser.add_argument("--eps", type=float, default=1e-3, help="minimum gap to be considered as an improvement")
    parser.add_argument("--iter_cmp", type=int, default=10, help="number of iterations to compare improvement")
    parser.add_argument("--early_stop_pct", type=float, default=0, help="stop if the percentage of improved instances in the last <iter_cmp> is less than this")
    parser.add_argument("--max_iter", type=int, default=100, help="maximum number of iterations")    

    args = parser.parse_args()

    curr_dir = os.path.dirname(os.path.abspath(__file__))
    
    save_fn = get_save_fn(args)
    save_fn = os.path.join(curr_dir, "shpp_eval", get_save_fn(args))
    if os.path.exists(save_fn):
        print(f"File {save_fn} exists. Skip.")
        sys.exit(0)  
    
    
    
    model = os.path.join(curr_dir, f"pretrained/{args.model}")
    model_init = os.path.join(curr_dir, f"pretrained/{args.model_init}") if args.model_init is not None else model
    assert os.path.exists(model), f"Model {model} does not exist"
    assert os.path.exists(model_init), f"Model {model_init} does not exist" 
    
    data = load_data(args.n, seed=args.seed, I=args.I)
    
    try:
        ref = load_concorde_ref(args.n, seed=args.seed)
        ref_cost = ref["obj"][:args.I]
        print("Ground truth: {:.3f}+-{:.3f}".format(ref_cost.mean(), ref_cost.std()))
    except:
        print("Ground truth: not available")


    best, score_hist = eval_shpp_refine(
        model=model, data=data, n_splits=args.n_splits, model_init=model_init,
        eps=args.eps, iter_cmp=args.iter_cmp, early_stop_pct=args.early_stop_pct, max_iter=args.max_iter
    )


    res = copy.deepcopy(best)
    res["score_hist"] = score_hist
    res["args"] = args

    # dump the result
    with open(save_fn, "wb") as f:
        pickle.dump(res, f)
