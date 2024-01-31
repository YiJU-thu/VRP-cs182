import numpy as np
import pickle
import os, sys
import argparse
from loguru import logger

curr_dir = os.path.dirname(__file__)
utils_dir = os.path.join(curr_dir, "../utils_project")
if utils_dir not in sys.path:
    sys.path.append(utils_dir)
from utils_func import cprint, cstring
from utils_vrp import get_random_graph_np









def save_random_dataset(num_graphs, graph_size, rescale, force_triangle_iter, seed, problem="tsp", save=True, save_path=None):

    if save_path is not None and os.path.exists(save_path):
        logger.info(f"File {cstring.green(save_path)} already exists!")
        with open(save_path, "rb") as f:
            dumped = pickle.load(f)
        return dumped

    data = get_random_graph_np(n=graph_size, num_graphs=num_graphs, non_Euc=True, 
                               rescale=rescale, force_triangle_iter=force_triangle_iter, seed=seed,
                               is_cvrp=(problem=="cvrp"), )
    # data has keys "coords", "rel_distance", "distance", "scale_factors"
    if save:
        assert save_path is not None
        with open(save_path, "wb") as f:
            pickle.dump(data, f)
        logger.success(f"Saved dataset to {cstring.green(save_path)}")
    return data

def save_mini_copy(data, sample_size=100, save=True, save_path=None):
    if save_path is not None and os.path.exists(save_path):
        logger.info(f"File {cstring.green(save_path)} already exists!")
        with open(save_path, "rb") as f:
            dumped = pickle.load(f)
        return dumped
    
    for key in data:
        if data[key] is not None:
            data[key] = data[key][:sample_size]
        else:
            data[key] = None
    if save:
        assert save_path is not None
        with open(save_path, "wb") as f:
            pickle.dump(data, f)
        logger.success(f"{cstring.blue('mini copy')} Saved dataset to {cstring.green(save_path)}")
    return data
    


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--problem", type=str, default="tsp", help="problem type. tsp or cvrp")
    parser.add_argument("--num_graphs", type=int, default=1000, help="number of samples")
    parser.add_argument("--graph_size", type=int, default=20, help="number of nodes")
    parser.add_argument("--rescale", action="store_true", help="whether to include scale factors")
    parser.add_argument("--force_triangle_iter", type=int, default=0, help="force triangle inequality for this many iterations")
    parser.add_argument("--save_dir", type=str, default=None, help="save directory")
    parser.add_argument("--mini_copy", action="store_true", help="save a mini copy of the dataset")
    # parser.add_argument("--save_mini_dir", type=str, default=None, help="save directory of mini samples")
    parser.add_argument("--seed", type=int, default=1234, help="random seed")
    args = parser.parse_args()
    
    assert args.problem in ["tsp", "cvrp"], "problem type should be tsp or cvrp"

    # TODO: make sure you are not saying the dataset under VRP-CS182 (too large!)
    gitrepo_path = os.path.abspath(os.path.join(curr_dir, "../"))
    assert args.save_dir is not None
    save_dir = os.path.abspath(args.save_dir)
    in_gitrepo = save_dir.startswith(gitrepo_path + os.sep)    
    # if the save path is in the git repo, we add "NoTrack" in the name to avoid tracking

    dist_tag = "C" if args.rescale else "S"
    save_fn = f"rnd_N{args.graph_size}_I{args.num_graphs}_{dist_tag}_seed{args.seed}_iter{args.force_triangle_iter}"
    if args.problem == "cvrp":
        save_fn = f"CVRP_{save_fn}"
    save_path = os.path.join(args.save_dir, save_fn+("_NoTrack"*in_gitrepo)+".pkl")
    data = save_random_dataset(num_graphs=args.num_graphs, graph_size=args.graph_size, 
                               rescale=args.rescale, force_triangle_iter=args.force_triangle_iter, 
                               seed=args.seed, save=True, save_path=save_path, problem=args.problem)
    
    if args.mini_copy:
        mini_size=100
        mini_fn = f"{save_fn}_mini{mini_size}.pkl"
        save_mini_dir = os.path.join(curr_dir, "../dataset")
        mini_path = os.path.join(save_mini_dir, mini_fn)
        save_mini_copy(data, sample_size=mini_size, save=True, save_path=mini_path)

    


