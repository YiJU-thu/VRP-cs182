import os, sys
import numpy as np
import torch
import argparse
import pickle

from eval import eval_dataset, _eval_dataset
from options import get_eval_options, get_options

curr_dir = os.path.dirname(__file__)
util_path = os.path.join(curr_dir, '..', 'utils_project')
if util_path not in sys.path:
    sys.path.append(util_path)
from utils_vrp import get_tour_len_torch, to_torch, to_np

from loguru import logger
import pandas as pd
import numpy as np
import json
import time


def _tour_zeropad(tours):
    # pad tours to the same length (solution from CVRP)
    max_len = max([len(t) for t in tours])
    tours_arr = np.array([np.pad(t, (0, max_len - len(t)), mode='constant').astype(int) for t in tours])
    return tours_arr

@logger.catch
def eval_nE_tsp(model, dataset, recompute_cost=True,
                decode_strategy='greedy', width=0, gamma=0, max_calc_batch_size=10000, 
                eas_layer=None, eas_batch_size=None, max_runtime_per_instance=0.05):

    # suggest always use recompute, since tsp.make_dataset has normalized the dataset

    decode_strategy = decode_strategy
    width = width
    
    if eas_layer is not None:
        eval_batch_size = eas_batch_size
    else:
        eval_batch_size = max(1, max_calc_batch_size // (width if width > 0 else 1))
    logger.info(f"batch_size = {eval_batch_size}")

    cmd = ["--datasets", 'None',
            "--model", model,
            "--decode_strategy", decode_strategy,
            "--width", str(width),
            "--eval_batch_size", str(int(eval_batch_size)),
            "--max_calc_batch_size", str(int(max_calc_batch_size)),
            "--gamma", str(gamma),
            "--eas_layer",  eas_layer,
            '--max_runtime_per_instance', str(max_runtime_per_instance)]

    opts = get_eval_options(cmd)
    t0 = time.perf_counter()
    costs, tours, durations = eval_dataset(dataset_path=None, dataset=dataset,
             width=width, softmax_temp=1, opts=opts)
    wall_time = time.perf_counter() - t0

    # NOTE: in cvrp, each tour may have different length
    costs = np.array(costs)
    min_len, max_len = min([len(t) for t in tours]), max([len(t) for t in tours])
    if min_len == max_len:
        tours = np.array(tours)
    else:
        tours = _tour_zeropad(tours)
    durations = np.array(durations)

    if not recompute_cost:
        return costs, tours, durations, wall_time
    
    dataset = to_torch(dataset)
    tours = torch.from_numpy(tours)
    costs = get_tour_len_torch(dataset, tours)
    return costs.cpu().numpy(), tours.cpu().numpy(), durations, wall_time

def _get_subset_I(dataset, I):
    if I is None:
        return dataset
    for key in dataset.keys():
        if isinstance(dataset[key], list):
            dataset[key] = dataset[key][:I]
        elif isinstance(dataset[key], np.ndarray):
            dataset[key] = dataset[key][:I]
    return dataset

def load_rnd_dataset(dataset_path, I, rnd_dist):
    assert rnd_dist in ['standard', 'complex']
    with open(dataset_path, 'rb') as f:
        dataset = pickle.load(f)

    dataset = _get_subset_I(dataset, I)
    
    assert "coords" in dataset.keys() or "distance" in dataset.keys()
    if rnd_dist == 'standard' and dataset.get('scale_factors') is not None:
        dataset['scale_factors'] = None
    elif rnd_dist == 'complex':
        assert dataset.get('scale_factors') is not None
    return dataset


def load_amz_dataset(dataset_path, I, to_np=False, skip_station=False):
    with open(dataset_path, 'rb') as f:
        dataset = pickle.load(f)
    assert "coords" in dataset.keys()
    assert "distance" in dataset.keys()
    
    dataset = _get_subset_I(dataset, I)

    if isinstance(dataset["coords"], list) and to_np:
        dataset["coords"] = np.array(dataset["coords"])
    if isinstance(dataset["distance"], list) and to_np:
        dataset["distance"] = np.array(dataset["distance"])
    
    dataset['scale_factors'] = None
    
    if not skip_station:
        return dataset
    assert "station" in dataset.keys(), "dataset should have key 'station'"

    coords = []
    distance = []

    for i in range(len(dataset["coords"])):
        n = len(dataset["coords"][i])
        s = dataset["station"][i]
        idx = np.concatenate([np.arange(s), np.arange(s+1,n)]).astype(int)
        coords.append(dataset["coords"][i][idx])
        distance.append(dataset["distance"][i][idx][:, idx])
    if to_np:
        coords = np.array(coords)
        distance = np.array(distance)
    dataset['coords'] = coords
    dataset['distance'] = distance

    # for key in list(dataset.keys()):
    #     if key not in ['coords', 'distance', 'scale_factors']:
    #         dataset.pop(key)
    logger.debug(f"Dataset keys: {dataset.keys()}")
    return dataset




def load_dataset(dataset_path, dataset_type, I=None, **kwargs):

    assert dataset_type in ['rnd', 'amz']

    if dataset_type == 'rnd':
        dataset = load_rnd_dataset(dataset_path, I=I, **kwargs)
    elif dataset_type == 'amz':
        dataset = load_amz_dataset(dataset_path, I=I, **kwargs)
    return to_torch(dataset)


@logger.catch
def run_eval(model, ds, data_dir, I=None, config=None):
    
    # FIXME: currently, test tasks are fixed ... 
    tsp_ds_names = {
        'amz_nS': 'amazon_N100_I1000_seed1234_S_NoTrack',
        'amz_S': 'amazon_N100_I1000_seed1234_S_NoTrack',
        'rnd_S': 'rnd_N100_I5000_S_seed0_iter4_NoTrack',
        'rnd_C': 'rnd_N100_I5000_C_seed0_iter4_NoTrack',
        'rnd_NC': 'rnd_N100_I10000_NC_seed0_iter4_NoTrack',
        'rnd_200': 'rnd_N200_I100_S_seed0_iter4_NoTrack',
        'rnd_500': 'rnd_N500_I100_S_seed0_iter4_NoTrack',
        'rnd_1000': 'rnd_N1000_I100_S_seed0_iter4_NoTrack',
    }

    cvrp_ds_names = {
        'rnd_S': 'CVRP_rnd_N100_I1000_S_seed0_iter4_NoTrack',
    }
    
    model_dir = os.path.dirname(model) if not os.path.isdir(model) else model
    model_args = os.path.join(model_dir, 'args.json')
    with open(model_args, 'r') as f:
        args = json.load(f)
    problem = args['problem']
    
    if problem == 'tsp':
        assert ds in tsp_ds_names.keys(), f"dataset {ds} not supported for TSP. Only support {tsp_ds_names.keys()}"
        dataset_path = os.path.join(data_dir, f"{tsp_ds_names[ds]}.pkl")
    elif problem == 'cvrp':
        assert ds in cvrp_ds_names.keys(), f"dataset {ds} not supported for CVRP. Only support {cvrp_ds_names.keys()}"
        dataset_path = os.path.join(data_dir, f"{cvrp_ds_names[ds]}.pkl")
    else:
        raise NotImplementedError(f"problem {problem} not implemented")


    if ds in ["rnd_200", "rnd_500", "rnd_1000"]:
        n = int(ds.split('_')[1])
    else:
        n = 100

    # load dataset on cpu
    with torch.device('cpu'):
        if ds == 'amz_nS' or ds == 'amz_S':
            dataset = load_dataset(dataset_path, dataset_type='amz', to_np=True, skip_station=(ds=='amz_nS'), I=I)
        else:
            dataset = load_dataset(dataset_path, dataset_type='rnd', rnd_dist='complex' if ds=='rnd_C' else 'standard', I=I)
    
    config = {} if config is None else config
    if "max_calc_batch_size" not in config:
        config["max_calc_batch_size"] = 10000 if n <= 100 else 1000
    config["max_runtime_per_instance"] = config["max_runtime_per_instance"] * n / 100
    
    costs, tours, durations, wall_time = eval_nE_tsp(model, dataset, recompute_cost=True, **config)
    res = {
        'obj': costs,
        'tour': tours,
        'time': durations,
        'wall_time': wall_time
    }
    return res



if __name__ == "__main__":
    
    logger.add('logs/eval.log')

    parser = argparse.ArgumentParser()
    parser.add_argument('--log_fn', type=str, default='eval_log_May20')
    parser.add_argument('--sol_dir', type=str, default='neurips24-may')
    parser.add_argument('--do_tag', type=str, default='W')
    parser.add_argument('--redo_failed', action='store_true')
    parser.add_argument('--I', type=int, default=None)
    parser.add_argument('--no_save', action='store_true')
    # parser.add_argument('--eval_batch_size', type=int, default=1000)
    parser.add_argument('--max_calc_batch_size', type=int, default=10000)    
    
    # these are used for debugging one model on one dataset
    # please use --log_fn none, --no_save True
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--ds', type=str, default=None)
    parser.add_argument('--decode_strategy', type=str, default="greedy")
    parser.add_argument('--width', type=int, default=0)
    parser.add_argument('--gamma', type=int, default=0)
    parser.add_argument('--eas_layer', type=str, default=None)
    parser.add_argument('--max_runtime_per_instance', type=float, default=0.05)
    parser.add_argument('--eas_batch_size', type=int, default=1000)

    args = parser.parse_args()

    # convert "none", "None", "NONE" to None
    for k, v in vars(args).items():
        if v is not None and v in ["none", "None", "NONE"]:
            setattr(args, k, None)

    curr_dir = os.path.dirname(__file__)
    data_dir = os.path.join(curr_dir, '..', 'dataset/nE2024_data/test')
    model_dir = os.path.join(curr_dir, 'pretrained/neurips')
    
    if args.log_fn is None:
        assert args.model is not None and args.ds is not None
        config = {"decode_strategy": args.decode_strategy, "width": args.width,
                  "gamma": args.gamma, "max_calc_batch_size": args.max_calc_batch_size,
                  "eas_layer": args.eas_layer, "eas_batch_size": args.eas_batch_size,
                  "max_runtime_per_instance": args.max_runtime_per_instance}
        model_path = os.path.join(model_dir, args.model)
        res = run_eval(model = model_path, ds = args.ds, data_dir = data_dir, I=args.I, 
                       config = config)

        logger.info(f"costs: {res['obj'].mean():.3f}+-{res['obj'].std():.3f}. Time: {res['wall_time']:.3f}")
        # exit
        exit()
    

    # FIXME: WIP
    log_fn = f'{args.log_fn}.xlsx'
    # sheet names are decode_strategy
    # sheet colomns include:
    # model, epoch, (width, gamma), rnd_S, rnd_C, rnd_NC, amz_nS, rnd_S_T, ...

    # logic:
    # - read log_fn
    # - look for 'args.do_tag', e.g., 'W' (if redo_failed, look for corresponding fail_tag as well)
    # - find the first do_tag, record its row idx, replace it w/ corresponding run_tag, e.g., 'R', save to log_fn
    # - run the corresponding model on the corresponding dataset, if sol does not exist
    # - save results under folder model_dir/args.sol_dir w/ name incl. dataset, decode_strategy, width, gamma, etc.
    # - save results to log_fn

    

    def _get_tags(do_tag):
        # do_tag is 'W_xxx', then run_tag is 'R_xxx', fail_tag is 'F_xxx'
        run_tag, fail_tag = 'R' + do_tag[1:], 'F' + do_tag[1:]
        return run_tag, fail_tag
    
    def _get_sol_save_fn(ds, decode_strategy, width, gamma, eas_layer, eas_batch_size, tag=None):
        fn = f'{ds}_{decode_strategy}'
        if width > 0:
            fn += f'_w-{width}'
        if gamma > 0:
            fn += f'_g-{gamma}'
        if eas_layer is not None:
            fn += f'_eas-{eas_layer[0].upper()}-{eas_batch_size}'
        if tag is not None:
                    fn += f'_{tag}'
        return f'{fn}.pkl'

    def _find_first_job(df, do_tags):
        sig = 0
        for do_tag in do_tags:
            for i in range(len(df)):
                idx_W = np.where(df.iloc[i].values == do_tag)[0]
                if len(idx_W) > 0:
                    idx = idx_W[0]
                    sig = 1
                    return i, idx
        if sig == 0:
            
            return None, None     

    do_tag = args.do_tag
    run_tag, fail_tag = _get_tags(do_tag)
    
    decode_strategy = args.decode_strategy
    assert decode_strategy in ['greedy', 'sample', 'bs', 'sgbs', 'eas']


    if args.redo_failed:
        # replace all failed tags with do_tag
        dfs = pd.read_excel(log_fn, index_col=0, sheet_name=None)
        df = dfs[decode_strategy]
        for i in range(len(df)):
            for j in range(len(df.columns)):
                if df.iloc[i, j] == fail_tag:
                    df.iloc[i, j] = do_tag
        with pd.ExcelWriter(log_fn) as writer:
            for k, v in dfs.items():
                v.to_excel(writer, sheet_name=k)

    while True:
        dfs = pd.read_excel(log_fn, index_col=0, sheet_name=None)
        df = dfs[decode_strategy]
        i, idx = _find_first_job(df, do_tags=[do_tag])
        
        if i is None:
            logger.info(f"{decode_strategy}: All done!")
            break
        
        df.iloc[i, idx] = run_tag
        
        # write all files back to log_fn
        with pd.ExcelWriter(log_fn) as writer:
            for k, v in dfs.items():
                v.to_excel(writer, sheet_name=k)
        
        
        model = f"{df.index[i]}/epoch-{int(df.iloc[i]['epoch'])}.pt"
        model_path = os.path.join(model_dir, model)
        
        ds = df.columns[idx]

        width = int(df.iloc[i].get('width', 0))
        gamma = int(df.iloc[i].get('gamma', 0))
        eas_layer = df.iloc[i].get('layer', None)
        eas_batch_size = df.iloc[i].get('batch_size', 1000)

        tag = df.iloc[i].get('tag', None)
        if tag in ["T1", "T2"]:
            max_runtime_per_instance = 0.05 if tag == "T1" else 0.5
        else:
            max_runtime_per_instance = args.max_runtime_per_instance


        if decode_strategy == 'eas':
            _decode_strategy = df.iloc[i].get('strategy', 'greedy')
        else:
            _decode_strategy = decode_strategy

        config = {
            "decode_strategy": _decode_strategy,
            "width": width,
            "gamma": gamma,
            "max_calc_batch_size": args.max_calc_batch_size,
            "eas_layer": eas_layer,
            "eas_batch_size": eas_batch_size,
            "max_runtime_per_instance": max_runtime_per_instance,
        }


        res_fn = _get_sol_save_fn(ds, decode_strategy, width, gamma, eas_layer, eas_batch_size, tag)
        sol_dir = os.path.join(os.path.dirname(model_path), args.sol_dir)
        # FIXME: if exist, continue

        logger.info(f"Eval model {model} on dataset {ds}")
        
        try:
            res = run_eval(model_path, ds, data_dir, I=args.I, config=config)
            logger.info(f"costs: {res['obj'].mean():.3f}+-{res['obj'].std():.3f}. Time: {res['wall_time']:.3f}")
            
            os.makedirs(sol_dir, exist_ok=True)
            if not args.no_save:
                with open(os.path.join(sol_dir, res_fn), 'wb') as f:
                    pickle.dump(res, f)
            
            dfs = pd.read_excel(log_fn, index_col=0, sheet_name=None)   # reload, may be updated by other process
            df = dfs[decode_strategy]

            df.iloc[i, idx] = f"{res['obj'].mean():.3f}"
            if ds+'_T' not in df.columns:
                df[ds+'_T'] = ''
            # get column index of ds+'_T'
            df.iloc[i,df.columns.get_loc(ds+'_T')] = f"{res['wall_time']:.2f}"


            # write all files back to log_fn
            with pd.ExcelWriter(log_fn) as writer:
                for k, v in dfs.items():
                    v.to_excel(writer, sheet_name=k)
            logger.success(f"Eval model {model} on dataset {ds} done")

        except Exception as e:
            logger.error(f"Eval model {model} on dataset {ds} failed")
            logger.error(e)
            
            dfs = pd.read_excel(log_fn, index_col=0, sheet_name=None)   # reload, may be updated by other process
            df = dfs[decode_strategy]
            df.iloc[i, idx] = fail_tag
            # write all files back to log_fn
            with pd.ExcelWriter(log_fn) as writer:
                for k, v in dfs.items():
                    v.to_excel(writer, sheet_name=k)