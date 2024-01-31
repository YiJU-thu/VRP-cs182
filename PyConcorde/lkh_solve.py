import os, sys
import pickle
from subprocess import check_call
import numpy as np
import time
from loguru import logger

@logger.catch
def solve_lkh(directory, name, dist_mat, scale=10000, runs=1):
    executable = '/home/yiju/install/LKH-3.0.6/LKH'
    problem_filename = os.path.join(directory, "{}.lkh{}.vrp".format(name, runs))
    tour_filename = os.path.join(directory, "{}.lkh{}.tour".format(name, runs))
    # output_filename = os.path.join(directory, "{}.lkh{}.pkl".format(name, runs))
    param_filename = os.path.join(directory, "{}.lkh{}.par".format(name, runs))
    log_filename = os.path.join(directory, "{}.lkh{}.log".format(name, runs))

    # try:  NOTE: try & except is protected in concorde_solve.py
    #     # # May have already been run
    #     # if os.path.isfile(output_filename) and not disable_cache:
    #     #     tour, duration = load_dataset(output_filename)
    #     # else:
    
    write_atsp_vrp(problem_filename, dist_mat, scale=scale, name=name)

    params = {"PROBLEM_FILE": problem_filename, "OUTPUT_TOUR_FILE": tour_filename, "RUNS": runs, "SEED": 1234}
    write_lkh_par(param_filename, params)

    with open(log_filename, 'w') as f:
        start = time.time()
        check_call([executable, param_filename], stdout=f, stderr=f)
        duration = time.time() - start

    tour = read_tsplib(tour_filename)
    obj = dist_mat[tour, np.roll(tour,-1)].sum()

    # delete files
    for f in [problem_filename, tour_filename, param_filename, log_filename]:
        os.remove(f)

    return obj, tour, duration
    
    # except Exception as e:
    #     print("Exception occured")
    #     print(e)
    #     return None


def write_atsp_vrp(filename, dist_mat, scale=10000, name="problem"):
    n = len(dist_mat)
    with open(filename, 'w') as f:
        f.write("\n".join([
            "{} : {}".format(k, v)
            for k, v in (
                ("NAME", name),
                ("TYPE", "ATSP"),
                ("DIMENSION", n),
                ("EDGE_WEIGHT_TYPE", "EXPLICIT"),
                ("EDGE_WEIGHT_FORMAT", "FULL_MATRIX"),
            )
        ]))
        f.write("\n")
        f.write("EDGE_WEIGHT_SECTION\n")
        for i in range(n):
            f.write("\t".join([str(int(dist_mat[i, j]*scale+0.5)) for j in range(n)]))
            f.write("\n")
        f.write("EOF\n")


def read_tsplib(filename):
    with open(filename, 'r') as f:
        tour = []
        dimension = 0
        started = False
        for line in f:
            if started:
                loc = int(line)
                if loc == -1:
                    break
                tour.append(loc)
            if line.startswith("DIMENSION"):
                dimension = int(line.split(" ")[-1])

            if line.startswith("TOUR_SECTION"):
                started = True

    assert len(tour) == dimension
    tour = np.array(tour).astype(int) - 1  # Subtract 1 as depot is 1 and should be 0
    return tour


def write_lkh_par(filename, parameters):
    default_parameters = {  # Use none to include as flag instead of kv
        "MAX_TRIALS": 10000,
        "RUNS": 10,
        "TRACE_LEVEL": 1,
        "SEED": 0
    }
    with open(filename, 'w') as f:
        for k, v in {**default_parameters, **parameters}.items():
            if v is None:
                f.write("{}\n".format(k))
            else:
                f.write("{} = {}\n".format(k, v))