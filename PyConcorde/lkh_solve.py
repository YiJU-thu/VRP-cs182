import os, sys
import pickle
from subprocess import check_call
import numpy as np
import time
from loguru import logger

# @logger.catch
def solve_lkh(directory, name, dist_mat, demand=None, depot=None, capacity=None, scale=10000, runs=1, problem='tsp'):
    executable = '/home/yiju/install/LKH-3.0.6/LKH'
    problem_filename = os.path.join(directory, "{}.lkh{}.vrp".format(name, runs))
    tour_filename = os.path.join(directory, "{}.lkh{}.tour".format(name, runs))
    # output_filename = os.path.join(directory, "{}.lkh{}.pkl".format(name, runs))
    param_filename = os.path.join(directory, "{}.lkh{}.par".format(name, runs))
    log_filename = os.path.join(directory, "{}.lkh{}.log".format(name, runs))

    if problem == 'tsp':
        write_atsp_vrp(problem_filename, dist_mat, scale=scale, name=name)
    elif problem == 'cvrp':
        depot = 0 if depot is None else depot
        assert depot == 0, "Depot must be 0 for CVRP (not necessarily, but due to current implementation)"
        capacity = 1 if capacity is None else capacity # FIXME: this will almost always cause a issue, as demand must be integer
        assert demand is not None, "Demand is required for CVRP"
        write_acvrp(problem_filename, dist_mat, demand, depot=depot, capacity=capacity, scale=scale, name=name)
    else:
        raise KeyError("Unknown problem: {}".format(problem))

    params = {"PROBLEM_FILE": problem_filename, "OUTPUT_TOUR_FILE": tour_filename, "RUNS": runs, "SEED": 1234,
              "MAX_TRIALS": 10000 if problem == 'tsp' else 1000, "TIME_LIMIT": 600}
    write_lkh_par(param_filename, params)

    with open(log_filename, 'w') as f:
        start = time.time()
        check_call([executable, param_filename], stdout=f, stderr=f)
        duration = time.time() - start

    if problem == 'tsp':
        tour = read_tsplib(tour_filename)
        obj = dist_mat[tour, np.roll(tour,-1)].sum()
    elif problem == 'cvrp':
        tour = read_vrplib(tour_filename, len(dist_mat)-1)
        tour_inv = tour[::-1]
        obj = dist_mat[tour, np.roll(tour,-1)].sum()
        obj_inv = dist_mat[tour_inv, np.roll(tour_inv,-1)].sum()
        # TODO: check if obj_inv is better

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


def write_acvrp(filename, dist_mat, demand, depot=0, capacity=30, scale=10000, name="problem"):
    n = len(dist_mat)
    with open(filename, 'w') as f:
        f.write("\n".join([
            "{} : {}".format(k, v)
            for k, v in (
                ("NAME", name),
                ("TYPE", "CVRP"),
                # FIXME: we use CVRP, which only solves symmetric problems
                ("DIMENSION", n),
                ("EDGE_WEIGHT_TYPE", "EXPLICIT"),
                ("EDGE_WEIGHT_FORMAT", "FULL_MATRIX"),
                ('CAPACITY', capacity),
                ("DISPLAY_DATA_TYPE", "NO_DISPLAY"),
            )
        ]))
        f.write("\n")
        f.write("EDGE_WEIGHT_SECTION\n")
        dist_mat = dist_mat + np.eye(n)*1e8
        
        for i in range(n):
            f.write("\t".join([str(int(dist_mat[i, j]*scale+0.5)) for j in range(n)]))
            f.write("\n")
        
        demand = demand.tolist()
        demand.insert(depot,0)
        assert len(demand) == n
        
        f.write("DEMAND_SECTION\n")
        f.write("\n".join([
            "{}\t{}".format(i + 1, round(d*capacity))
            for i, d in enumerate(demand)
        ]))
        f.write("\n")
        f.write("DEPOT_SECTION\n")
        f.write(f"{depot+1}\n")
        f.write("EOF\n")


def read_vrplib(filename, n):
    # n (excl. depot) is needed as a tour may contain nodes above n (contain multiple routes)
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
    tour[tour > n] = 0  # Any nodes above the number of nodes there are is also depot
    assert tour[0] == 0  # Tour should start with depot
    assert tour[-1] != 0  # Tour should not end with depot
    return tour.tolist()    # include the first node (depot)




def write_lkh_par(filename, parameters):
    default_parameters = {  # Use none to include as flag instead of kv
        "MAX_TRIALS": 10000,
        "RUNS": 10,
        "TRACE_LEVEL": 1,
        "SEED": 0
    }
    default_parameters.update(parameters)
    with open(filename, 'w') as f:
        for k, v in {**default_parameters}.items():
            if v is None:
                f.write("{}\n".format(k))
            else:
                f.write("{} = {}\n".format(k, v))