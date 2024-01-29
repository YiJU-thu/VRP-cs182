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


@logger.catch
def eval_nE_tsp(model, dataset, recompute_cost=True,
                decode_strategy='greedy', width=0, max_calc_batch_size=10000):

    # suggest always use recompute, since tsp.make_dataset has normalized the dataset

    decode_strategy = decode_strategy
    width = width
    
    eval_batch_size = max_calc_batch_size // (width if width > 0 else 1)

    cmd = ["--datasets", 'None',
            "--model", model,
            "--decode_strategy", decode_strategy,
            "--width", str(width),
            "--eval_batch_size", str(eval_batch_size),
            "--max_calc_batch_size", str(max_calc_batch_size)]

    opts = get_eval_options(cmd)
    costs, tours, durations = eval_dataset(dataset_path=None, dataset=dataset,
             width=width, softmax_temp=1, opts=opts)
    
    costs = np.array(costs)
    tours = np.array(tours)
    durations = np.array(durations)

    if not recompute_cost:
        return costs, tours, durations
    
    dataset = to_torch(dataset)
    tours = torch.from_numpy(tours)
    costs = get_tour_len_torch(dataset, tours)
    return costs.cpu().numpy(), tours.cpu().numpy(), durations


def load_rnd_dataset(dataset_path, rnd_dist):
    assert rnd_dist in ['standard', 'complex']
    with open(dataset_path, 'rb') as f:
        dataset = pickle.load(f)
    assert "coords" in dataset.keys()
    assert "distance" in dataset.keys()
    if rnd_dist == 'standard' and dataset.get('scale_factors') is not None:
        dataset['scale_factors'] = None
    elif rnd_dist == 'complex':
        assert dataset.get('scale_factors') is not None
    return dataset


def load_amz_dataset(dataset_path, to_np=False, skip_station=False):
    with open(dataset_path, 'rb') as f:
        dataset = pickle.load(f)
    assert "coords" in dataset.keys()
    assert "distance" in dataset.keys()
    
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




def load_dataset(dataset_path, dataset_type, **kwargs):

    assert dataset_type in ['rnd', 'amz']

    if dataset_type == 'rnd':
        dataset = load_rnd_dataset(dataset_path, **kwargs)
    elif dataset_type == 'amz':
        dataset = load_amz_dataset(dataset_path, **kwargs)
    return to_torch(dataset)



def run_eval(model, ds, data_dir, config=None):
    assert ds in ['amz_nS', 'amz_S', 'rnd_S', 'rnd_C', 'amz_eval']
    n = int(model.split('tsp')[1].split('_')[0])
    seed = 1234
    if ds == 'amz_nS' or ds == 'amz_S':
        dataset_path = os.path.join(data_dir, f"amazon_eval_N{n}_I1000_seed1234_S.pkl")
        dataset = load_dataset(dataset_path, dataset_type='amz', to_np=True, skip_station=(ds=='amz_nS'))
    elif ds == 'rnd_S' or ds == 'rnd_C':
        dataset_path = os.path.join(data_dir, f"rnd_N{n}_I1000_{ds[-1]}_seed1234_iter4.pkl")
        dataset = load_dataset(dataset_path, dataset_type='rnd', rnd_dist='standard' if ds=='rnd_S' else 'complex')
    elif ds == 'amz_eval':
        raise NotImplementedError
    
    config = {} if config is None else config
    config["max_calc_batch_size"] = 10000 if n <= 100 else 1000
    costs, tours, durations = eval_nE_tsp(model, dataset, recompute_cost=True, **config)
    res = {
        'obj': costs,
        'tour': tours,
        'time': durations
    }
    return res



if __name__ == "__main__":
    
    logger.add('logs/eval.log')
    
    data_dir = "/home/ecal_team/datasets/amazon_vrp/amazon_processed"
    
    log_fn = 'am_eval_log_Jan26.xlsx'
    
    while True:
        df = pd.read_excel(log_fn, index_col=0)
        sig = 0
        if 'W' not in df.values:
            logger.info("All done!")
            break

        for i in range(len(df)):
            idx_W = np.where(df.iloc[i].values == 'W')[0]
            if len(idx_W) > 0:
                idx = idx_W[0]
                df.iloc[i, idx] = 'R'
                df.to_excel(log_fn)
                sig = 1
                break
        
        if sig == 0:
            logger.info("All done!")
            break
        
        model = os.path.join('pretrained/Jan26-icml',df.index[i])
        ds = df.columns[idx]

        decode_strategy = df.iloc[i]['decode_strategy']
        width = int(df.iloc[i]['width'])
        config = {
            "decode_strategy": decode_strategy,
            "width": width
        }

        logger.info(f"Eval model {model} on dataset {ds}")
        
        try:
            res = run_eval(model, ds, data_dir, config)
            logger.success(f"Eval model {model} on dataset {ds} done")
            logger.info(f"costs: {res['obj'].mean():.3f}+-{res['obj'].std():.3f}")

            res_fn = f'Res_D_{ds}_{decode_strategy}-{width}.pkl'
            with open(os.path.join(model, res_fn), 'wb') as f:
                pickle.dump(res, f)
            df.iloc[i, idx] = res['obj'].mean()
            df.to_excel(log_fn)
        except Exception as e:
            logger.error(f"Eval model {model} on dataset {ds} failed")
            logger.error(e)
            df.iloc[i, idx] = 'F'
            df.to_excel(log_fn)