import os
import torch
import argparse
from loguru import logger
import shutil


def find_max_epoch(run_path):
    epoch_list = os.listdir(run_path)
    epochs = [fn for fn in epoch_list if fn.endswith(".pt")]
    if len(epochs) == 0:
        return None
    max_epoch = max([int(epoch.split("-")[1].split(".")[0]) for epoch in epochs])
    return max_epoch



def clear_checkpoint_optimizer_states(dir="outputs"):
    """
    visit all the subfolders in dir, if it is not the last epoch, only keep "model" in .pt file
    """
    # curr_dir = os.absdir(__file__)
    # os.debug(f"curr_dir: {curr_dir}")

    project_list = os.listdir(dir)
    for project in project_list:
        project_path = os.path.join(dir, project)
        if not os.path.isdir(project_path):
            continue
        run_list = os.listdir(project_path)
        for run in run_list:
            run_path = os.path.join(project_path, run)
            if not os.path.isdir(run_path):
                continue
            epoch_list = os.listdir(run_path)
            epochs = [fn for fn in epoch_list if fn.endswith(".pt")]
            max_epoch = find_max_epoch(run_path)
            if max_epoch is None:
                continue
            for e in epochs:
                epoch_path = os.path.join(run_path, e)
                if int(e.split("-")[1].split(".")[0]) < max_epoch:
                    logger.debug(f"clear {epoch_path}")
                    model = torch.load(epoch_path)
                    model_small = {"model": model["model"]}
                    torch.save(model_small, epoch_path)
                    logger.success(f"clear {run} {e}")
            logger.success(f"!! clear {run}")


def copy_trained_nets(from_dir, to_dir, epoch=None):
    """
    copy the trained nets from from_dir to to_dir
    """
    assert os.path.isdir(from_dir), f"{from_dir} is not a directory"
    # the parent directory of to_dir should exist
    to_dir_parent = os.path.dirname(to_dir)
    assert os.path.isdir(to_dir_parent), f"{to_dir_parent} is not a directory"
    
    if epoch is None:
        max_epoch = find_max_epoch(from_dir)
        if max_epoch is None:
            logger.error(f"no epoch in {from_dir}")
            return
        epoch_fn = f"epoch-{max_epoch}.pt"
    else:
        epoch_fn = f"epoch-{epoch}.pt"
    
    if os.path.isfile(os.path.join(to_dir, "args.json")) and os.path.isfile(os.path.join(to_dir, epoch_fn)):
        logger.info(f"{to_dir} exists")
        return
    
    if not os.path.isdir(to_dir):
        os.makedirs(to_dir)
    
    args_fn = os.path.join(from_dir, "args.json")
    assert os.path.isfile(args_fn), f"{args_fn} does not exist"
    # copy args.json to to_dir
    shutil.copy(args_fn, to_dir)

    epoch_path = os.path.join(from_dir, epoch_fn)
    assert os.path.isfile(epoch_path), f"{epoch_path} does not exist"
    # copy epoch-xx.pt to to_dir
    model = torch.load(epoch_path)
    model_small = {"model": model["model"]}
    torch.save(model_small, os.path.join(to_dir, epoch_fn))
    logger.success(f"copy {from_dir} to {to_dir}")
    
    return




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, default="AM2019/outputs")
    args = parser.parse_args()
    clear_checkpoint_optimizer_states(dir=args.dir)