from concorde.tsp import TSPSolver
from concorde.tests.data_utils import get_dataset_path
from concorde.problem import Problem
import numpy as np
from time import perf_counter
import time
import os, sys
import pickle
from loguru import logger
from time import sleep

_curr_dir = os.path.dirname(__file__)
utils_dir = os.path.join(_curr_dir, '../utils_project')
if utils_dir not in sys.path:
    sys.path.append(utils_dir)
from utils_vrp import get_tour_len



def clear_concorde_files():
    for fn in os.listdir():
        # end with .pul, .sav, .res, .tsp
        if fn.endswith(".sav") or fn.endswith(".pul")\
             or fn.endswith(".res") or fn.endswith(".tsp") or fn.endswith(".sol"):
            os.remove(fn)
        # end with *.[0-9][0-9][0-9]
        elif fn[-3:].isdigit() and fn[-4] == ".":
            os.remove(fn)


def concorde_atsp_matrix(dist_mat, scale=1, inf_M=1e7, big_M=1):
    
    n = dist_mat.shape[0]
    C_bar = dist_mat*scale - np.eye(n) * big_M    # use the notation in (Jonker & Volgenat, 1983)

    extend_mat = np.zeros((2*n, 2*n))
    extend_mat[:n, :n] = inf_M
    extend_mat[n:, n:] = inf_M
    extend_mat[n:, :n] = C_bar
    extend_mat[:n, n:] = C_bar.T
    
    return extend_mat


@logger.catch
def concorde_atsp_2d(dist_mat, scale=1, big_M=100, inf_M=1e7, tmp_idx=""):
    
    N = dist_mat.shape[0]

    extend_mat = concorde_atsp_matrix(dist_mat, scale=scale, big_M=big_M, inf_M=inf_M)
    extend_mat = np.round(extend_mat)   # concorde requires all distance be integers

    prob = Problem.from_matrix(extend_mat.astype(np.int32))
    
    # FIXME: this is wired, but the current pyconcorde do not support solve it directly
    tmp_fn = "tsp_NoTrack_test" + tmp_idx + ".tsp"
    prob.to_tsp(tmp_fn)
    solver = TSPSolver.from_tspfile(tmp_fn)
    
    t0 = perf_counter()
    sol = solver.solve(verbose=False)
    t = perf_counter() - t0
    obj = sol.optimal_value
    route = sol.tour

    route = route%N
    route1 = route[::2]
    route2 = route[1::2]
    assert np.all(route1 == route2) or np.all(route1==np.roll(route2, 1)), \
        "invalid route - increase big_M"
    if np.all(route1 == np.roll(route2, 1)):
        route1 = route1[::-1]   # FIXME: I do not know why, but it works???
    
    obj_actual = get_tour_len(tour=route1,dist_mat=dist_mat)
    # obj_actual should be similar to (obj+big_M*N)/scale
    obj_est = (obj + big_M*N) / scale
    eps = 1e-2 # 1%
    err = abs(obj_actual - obj_est) / obj_actual
    if err > eps:
        logger.warning(f"obj est. error {err:.2%} too large - increase scale")

    os.remove(tmp_fn)
    return obj_actual, route1, t


@logger.catch
def concorde_euc_2d(coords, scale=1):
    xs, ys = coords[:,0], coords[:,1]
    xs, ys = xs * scale, ys * scale
    solver = TSPSolver.from_data(xs=xs, ys=ys, norm="EUC_2D")
    t0 = perf_counter()
    sol = solver.solve(verbose=False)
    t = perf_counter() - t0
    obj = sol.optimal_value
    route = sol.tour

    obj_actual = get_tour_len(tour=route, coords=coords, norm="L2")
    # obj_actual should be similar to (obj+big_M*N)/scale
    obj_est = obj / scale
    eps = 1e-2 # 1%
    err = abs(obj_actual - obj_est) / obj_actual
    if err > eps:
        logger.warning(f"obj est. error {err:.2%} too large - increase scale")

    return obj_actual, route, t


def solve_one_instance(instance, type="EUC_2D", info="", **kws):
    try:
        if type == "EUC_2D":
            opt_value, route, t = concorde_euc_2d(instance, **kws)
        elif type == "ATSP":
            opt_value, route, t = concorde_atsp_2d(instance, **kws)
        logger.success(f"{info} | opt_value: {opt_value:.2f}, t: {t:.2f}")
    except:
        logger.error(f"{info} |", "failed")
        opt_value, route, t = np.inf, None, -np.inf
    return opt_value, route, t


def retrieve_res(out_fn, redo_failed=False):
    if not os.path.exists(out_fn):
        return None
    with open(out_fn, "rb") as f:
        res = pickle.load(f)
    # res should have keys: obj, time, tour, 
    # obj = -np.inf means error in solving
    # obj = np.inf means not solved yet
    if redo_failed:
        to_do_idx = np.where((res["obj"] == -np.inf) | (res["obj"] == np.inf))[0]
    else:
        to_do_idx = np.where(res["obj"] == np.inf)[0]
    return res, to_do_idx


def log_stats(data):
    def calc_stats(x):
        return np.mean(x), np.std(x), np.percentile(x, 5), np.percentile(x, 95)
    
    solved = np.where(data["time"] != -np.inf)[0]
    t_stats = calc_stats(data["time"][solved])
    obj_stats = calc_stats(data["obj"][solved])
    logger.info(f"{'='*20} [{len(solved):^4}] Solved {'='*20}")
    logger.info(f"time: {t_stats[0]:.3f}+-{t_stats[1]:.3f} (5%: {t_stats[2]:.3f}, 95%: {t_stats[3]:.3f})")
    logger.info(f"obj: {obj_stats[0]:.1f}+-{obj_stats[1]:.1f} (5%: {obj_stats[2]:.3f}, 95%: {obj_stats[3]:.3f})")
    logger.info("="*55)


if __name__ == "__main__":
    
    # sample:
    # python concorde_solve.py --type ATSP --I 100 --data <filename> --redo_failed --data_dir <data_dir> --out_dir <out_dir> --out_prefix <...> --clear 1 --save 1 --log <...>
    
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--type', type=str, default="ATSP", help='problem type. avail: ATSP, EUC_2D')
    parser.add_argument('--I', type=int, default=None, help='solve the first I unsolved problems')
    parser.add_argument('--redo_failed', action='store_true', help='redo failed instances')
    parser.add_argument('--data', type=str, help='data filename')
    parser.add_argument('--data_dir', type=str, default=None, help='data directory. use /dataset if None')
    parser.add_argument('--scale', type=float, default=1e1, help='scale the distance matrix / coords')  # FIXME: may auto find the right scale!
    parser.add_argument('--big_M', type=float, default=1e2, help='big M in ATSP')
    parser.add_argument('--out_dir', type=str, default=None, help='output directory. use current dir if None')
    parser.add_argument('--out_prefix', type=str, default="TSP_concorde", help='output prefix')
    parser.add_argument('--clear', type=int, default=100, help='clear .res files every CLEAR instances')
    parser.add_argument('--save', type=int, default=100, help='save results every SAVE instances')
    parser.add_argument('--log', type=str, default="concorde_log.log", help='log filename')
    
    args = parser.parse_args()
    
    data_dir = os.path.join(_curr_dir, '../dataset')
    data_dir = data_dir if args.data_dir is None else args.data_dir
    assert os.path.exists(data_dir), f"{data_dir} does not exist"
    data_fn = os.path.join(data_dir, args.data)
    assert os.path.exists(data_fn), f"{data_fn} does not exist"
    with open(data_fn, "rb") as f:
        data = pickle.load(f)

    # if EUC_2D, data.shape = (I,n,2)
    # if ATSP, data is a dict with keys: "distance", "coords", ("station", "acc", ...)
    assert args.type in ["ATSP", "EUC_2D"], "Invalid problem type, support ATSP and EUC_2D"
    if args.type == "ATSP":
        I_tot = len(data["distance"])   # number of instances, it may be a list (each instance different size), or a np.ndarray w/ shape (I,N,N)
    if args.type == "EUC_2D":
        I_tot = len(data)
    
    
    out_dir = _curr_dir if args.out_dir is None else args.out_dir
    assert os.path.exists(out_dir), f"{out_dir} does not exist"

    if args.log is not None:
        logger.add(args.log)
    
    
    out_fn = os.path.join(out_dir, f"{args.out_prefix}_{args.data}") # FIXME: data is .pkl, but this may be risky
    exist_res = retrieve_res(out_fn, redo_failed=args.redo_failed)
    if exist_res is not None:
        res, to_do_idx = exist_res
    else:
        res = {
            "obj": np.ones(I_tot) * (-np.inf),
            "time": np.ones(I_tot) * (-np.inf),
            "tour": [None] * I_tot
        }
        to_do_idx = np.arange(I_tot)
    
    to_do_idx = to_do_idx.astype(int)

    # TODO: the first I of to_do_idx
    I = len(to_do_idx) if args.I is None else min(len(to_do_idx), args.I)

    logger.info(f"Start solving {args.type} instances in {data_fn}, save to {out_dir}")
    logger.info(args)


    for i in range(I):
        idx = to_do_idx[i]
        
        if args.type == "ATSP": 
            opt_value, route, t = solve_one_instance(instance=data["distance"][idx], type="ATSP", info=f"id={idx}", scale=args.scale, big_M=args.big_M)
        elif args.type == "EUC_2D":
            opt_value, route, t = solve_one_instance(instance=data[idx], type="EUC_2D", info=f"id={idx}", scale=args.scale)
        
        res["obj"][idx] = opt_value
        res["time"][idx] = t
        res["tour"][idx] = route

        if (i+1) % args.clear == 0:
            clear_concorde_files()
        if (i+1) % args.save == 0:
            with open(out_fn, "wb") as f:
                pickle.dump(res, f)
            log_stats(res)
    
    sleep(1)
    clear_concorde_files()
    with open(out_fn, "wb") as f:
        pickle.dump(res, f)
    log_stats(res)



    