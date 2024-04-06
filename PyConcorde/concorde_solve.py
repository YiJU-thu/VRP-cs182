from concorde.tsp import TSPSolver
from concorde.tests.data_utils import get_dataset_path
from concorde.problem import Problem

from lkh_solve import solve_lkh

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
from utils_vrp import get_tour_len, recover_graph_np



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


# @logger.catch
def concorde_atsp_2d(dist_mat, scale=1, big_M=100, inf_M=1e7, tmp_idx="", time_bound=-1):
    from loguru import logger # this works for dask parallel

    N = dist_mat.shape[0]

    extend_mat = concorde_atsp_matrix(dist_mat, scale=scale, big_M=big_M, inf_M=inf_M)
    extend_mat = np.round(extend_mat)   # concorde requires all distance be integers

    prob = Problem.from_matrix(extend_mat.astype(np.int32))
    
    # FIXME: this is wired, but the current pyconcorde do not support solve it directly
    tmp_fn = "tsp_NoTrack_" + tmp_idx + ".tsp"
    prob.to_tsp(tmp_fn)
    solver = TSPSolver.from_tspfile(tmp_fn)
    
    t0 = perf_counter()
    sol = solver.solve(verbose=False, time_bound=time_bound)
    t = perf_counter() - t0
    obj = sol.optimal_value
    route = sol.tour
    assert not sol.hit_timebound, f"time bound hit: {time_bound} sec"

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


# @logger.catch
def concorde_euc_2d(coords, scale=1, time_bound=-1):
    from loguru import logger # this works for dask parallel
    xs, ys = coords[:,0], coords[:,1]
    xs, ys = xs * scale, ys * scale
    solver = TSPSolver.from_data(xs=xs, ys=ys, norm="EUC_2D")
    t0 = perf_counter()
    sol = solver.solve(verbose=False, time_bound=time_bound)
    t = perf_counter() - t0
    obj = sol.optimal_value
    route = sol.tour
    assert not sol.hit_timebound, f"time bound hit: {time_bound} sec"

    obj_actual = get_tour_len(tour=route, coords=coords, norm="L2")
    # obj_actual should be similar to (obj+big_M*N)/scale
    obj_est = obj / scale
    eps = 1e-2 # 1%
    err = abs(obj_actual - obj_est) / obj_actual
    if err > eps:
        logger.warning(f"obj est. error {err:.2%} too large - increase scale")

    return obj_actual, route, t


def solve_one_instance(instance, type="EUC_2D", solver="concorde", info="", problem="tsp", args=None, **kws):
    from loguru import logger # this works for dask parallel
    try:
        problem = problem
        if not (solver=="lkh" and type=="ATSP"):
            problem == "tsp", "CVRP is supported for ATSP AND lkh solver now"
        if type == "EUC_2D":
            opt_value, route, t = concorde_euc_2d(instance, **kws)
        elif type == "ATSP":
            if solver == "concorde":
                idx = info.split("=")[1]
                opt_value, route, t = concorde_atsp_2d(instance['distance'], tmp_idx=idx, **kws)
            elif solver == "lkh":   # FIXME: can write in a more elegant way
                directory = args.tmp_log_dir
                if not os.path.exists(directory):
                    os.makedirs(directory)
                name, runs = info, 1
                if problem == "tsp":
                    opt_value, route, t = solve_lkh(directory=directory, name=name, dist_mat=instance['distance'], 
                                                    runs=runs, scale=kws["scale"], problem=problem)
                elif problem == "cvrp":
                    opt_value, route, t = solve_lkh(directory=directory, name=name, dist_mat=instance['distance'], 
                                                    demand=instance['demand'], depot=0, capacity=args.capacity, 
                                                    runs=runs, scale=kws["scale"], problem=problem)
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
    if len(solved) == 0:
        logger.info("No solved instances")
        return
    t_stats = calc_stats(data["time"][solved])
    obj_stats = calc_stats(data["obj"][solved])
    logger.info(f"{'='*20} [{len(solved):^4}] Solved {'='*20}")
    logger.info(f"time: {t_stats[0]:.3f}+-{t_stats[1]:.3f} (5%: {t_stats[2]:.3f}, 95%: {t_stats[3]:.3f})")
    logger.info(f"obj: {obj_stats[0]:.3f}+-{obj_stats[1]:.3f} (5%: {obj_stats[2]:.3f}, 95%: {obj_stats[3]:.3f})")
    logger.info("="*55)


if __name__ == "__main__":
    
    # sample:
    # python concorde_solve.py --type ATSP --I 100 --data <filename> --redo_failed --data_dir <data_dir> --out_dir <out_dir> --out_prefix <...> --clear 1 --save 1 --log <...>
    
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--solver', type=str, default="concorde", help='solver name. avail: concorde, lkh')
    parser.add_argument('--problem', type=str, default="tsp", help='problem name. avail: tsp, cvrp')
    parser.add_argument('--type', type=str, default="ATSP", help='problem type. avail: ATSP, EUC_2D')
    parser.add_argument('--I', type=int, default=None, help='solve the first I unsolved problems')
    parser.add_argument('--redo_failed', action='store_true', help='redo failed instances')
    parser.add_argument('--data', type=str, help='data filename')
    parser.add_argument('--data_dir', type=str, default=None, help='data directory. use /dataset if None')
    parser.add_argument('--recover_graph', action='store_true', help='recover graph from scale_factors')
    parser.add_argument('--skip_station', action='store_true', help='skip station (depot) nodes in the data')
    parser.add_argument('--scale', type=float, default=1e1, help='scale the distance matrix / coords')  # FIXME: may auto find the right scale!
    parser.add_argument('--big_M', type=float, default=1e2, help='big M in ATSP')
    parser.add_argument('--capacity', type=int, default=None, help='capacity for CVRP')
    parser.add_argument('--time_bound', type=float, default=-1, help='time bound for each instance, default -1 means no time bound')
    parser.add_argument('--out_dir', type=str, default=None, help='output directory. use current dir if None')
    parser.add_argument('--out_prefix', type=str, default="TSP_concorde", help='output prefix')
    parser.add_argument('--clear', type=int, default=100, help='clear .res files every CLEAR instances')
    parser.add_argument('--save', type=int, default=100, help='save results every SAVE instances')
    parser.add_argument('--log', type=str, default="concorde_log.log", help='log filename')
    parser.add_argument('--tmp_log_dir', type=str, default="log_NoTrack", help='tmp log directory that stores intermediate log files')
    parser.add_argument('--ignore_NoTrack', action='store_true', help='remove NoTrack in file names so that the solutions will be tracked')
    parser.add_argument('--dask_parallel', action='store_true', help='use dask parallel to solve instances')
    parser.add_argument('--n_workers', type=int, default=None, help='number of workers in dask parallel')

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
    if args.recover_graph:
        data = recover_graph_np(data)
    
    
    if args.skip_station:
        assert "station" in data.keys(), "dataset should have key 'station'"
        coords = []
        distance = []
        for i in range(len(data["coords"])):
            n = len(data["coords"][i])
            s = data["station"][i]
            idx = np.concatenate([np.arange(s), np.arange(s+1,n)]).astype(int)
            coords.append(data["coords"][i][idx])
            distance.append(data["distance"][i][idx][:, idx])
        data['coords'] = coords
        data['distance'] = distance



    out_dir = _curr_dir if args.out_dir is None else args.out_dir
    assert os.path.exists(out_dir), f"{out_dir} does not exist"

    if args.log is not None:
        logger.add(args.log)
    
    
    out_fn = f"{args.out_prefix}_{args.data}"
    if args.ignore_NoTrack and "NoTrack" in out_fn:
        out_fn = out_fn.replace("_NoTrack", "")
    out_fn = os.path.join(out_dir, out_fn) # FIXME: data is .pkl, but this may be risky
    
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

    def get_data_item(data, idx):
        if isinstance(data, dict):
            return {k: get_data_item(v,idx) for k, v in data.items()}
        elif data is None:
            return None
        else:   # tuple, list, ndarray ...
            return data[idx]

    if args.dask_parallel:
        from dask.distributed import Client

        address = os.getenv("SCHED")    # HPC: get the scheduler address
        address = address + ":8786" if address is not None else None

        client = Client(address=address, n_workers=args.n_workers)
        logger.info(client)
        n_cores = sum(v for k, v in client.ncores().items())
        n_jobs = n_cores * args.save
        logger.info(f"n_jobs: {n_jobs}")
        
        i = 0

        while i < I:
            idxs = to_do_idx[i:min(I, i+n_jobs)]
            instances = [get_data_item(data, idx) for idx in idxs]
            if args.type == "ATSP":
                futures = [client.submit(solve_one_instance, instances[j], type="ATSP", problem=args.problem,
                        info=f"id={idxs[j]}", scale=args.scale, big_M=args.big_M, time_bound=args.time_bound, solver=args.solver, args=args) for j in range(len(idxs))]
            elif args.type == "EUC_2D":
                assert args.solver == "concorde", "Now only concorde support EUC_2D"
                futures = client.map(solve_one_instance, instances, type="EUC_2D", problem=args.problem,
                        info=[f"id={idx}" for idx in idxs], scale=args.scale, time_bound=args.time_bound, solver=args.solver, args=args)
            res_lst = client.gather(futures)
            for j, idx in enumerate(idxs):
                opt_value, route, t = res_lst[j]
                res["obj"][idx] = opt_value
                res["time"][idx] = t
                res["tour"][idx] = route
            if args.solver == "concorde":
                sleep(0.5)
                clear_concorde_files()
            with open(out_fn, "wb") as f:
                pickle.dump(res, f)
            log_stats(res)

            i += len(idxs)


    else:
        for i in range(I):
            
            idx = to_do_idx[i]
            instance = get_data_item(data, idx)
            if args.type == "ATSP": 
                opt_value, route, t = solve_one_instance(instance=instance, type="ATSP", problem=args.problem,
                        info=f"id={idx}", scale=args.scale, big_M=args.big_M, time_bound=args.time_bound, solver=args.solver, args=args)
            elif args.type == "EUC_2D":
                assert args.solver == "concorde", "Now only concorde support EUC_2D"
                opt_value, route, t = solve_one_instance(instance=instance, type="EUC_2D", problem=args.problem,
                        info=f"id={idx}", scale=args.scale, time_bound=args.time_bound, solver=args.solver, args=args)
            
            res["obj"][idx] = opt_value
            res["time"][idx] = t
            res["tour"][idx] = route

            if (i+1) % args.clear == 0 and args.solver == "concorde":
                sleep(0.5)
                clear_concorde_files()
            if (i+1) % args.save == 0:
                with open(out_fn, "wb") as f:
                    pickle.dump(res, f)
                log_stats(res)
            


    sleep(1)
    if args.solver == "concorde":
        clear_concorde_files()
    with open(out_fn, "wb") as f:
        pickle.dump(res, f)
    log_stats(res)



    