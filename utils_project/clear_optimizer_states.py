import os
import torch
import argparse
from loguru import logger
import shutil


def find_max_epoch(run_path):
    logger.warning("This function is deprecated, use find_default_epoch instead")
    return find_default_epoch(run_path, default_epoch="max")[0]

def find_default_epoch(run_path, default_epoch="best"):
    epoch_list = os.listdir(run_path)
    if default_epoch == "best":
        return "best", "best.pt"

    epochs = [fn for fn in epoch_list if fn.endswith(".pt")]
    if len(epochs) == 0:
        return None
    max_epoch = max([int(epoch.split("-")[1].split(".")[0]) for epoch in epochs])
    return max_epoch, f"epoch-{max_epoch}.pt"



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


def copy_trained_nets(from_dir, to_dir, epoches=None):
    """
    copy the trained nets from from_dir to to_dir
    """
    if not os.path.isdir(from_dir):
        from_dir, epoch_fn = os.path.split(from_dir)
        # epoch_fn: (1) "best.pt", (2) epoch-xxx.pt
        epoch = int(epoch_fn.split("-")[1].split(".")[0]) if epoch_fn != "best.pt" else "best"
        epoches = [epoch]
        epoch_fns = [epoch_fn]
    else:
        assert os.path.isdir(from_dir), f"{from_dir} is not a directory"
        if epoches is None:
            epoch, epoch_fn = find_default_epoch(from_dir, default_epoch)
            epoches, epoch_fns = [epoch], [epoch_fn]
            if epoch is None:
                logger.error(f"no epoch in {from_dir}")
                return
        else:
            epoch_fns = []
            for e in epoches:
                if isinstance(e, int):
                    epoch_fn = f"epoch-{e}.pt"
                else:
                    epoch, epoch_fn = find_default_epoch(from_dir, e)
                epoch_fns.append(epoch_fn)
            # epoch_fn = f"epoch-{epoch}.pt" if epoch != "best" else "best.pt"
            # epoch = int(epoch) if epoch != "best" else "best"
    
    if os.path.exists(os.path.join(to_dir, "args.json")) and os.path.exists(os.path.join(to_dir, epoch_fn)):
        logger.info(f"{to_dir} exists")
        return
    
    to_dir_parent = os.path.dirname(to_dir)     # the parent directory of to_dir should exist
    assert os.path.isdir(to_dir_parent), f"{to_dir_parent} is not a directory"
    if not os.path.isdir(to_dir):
        os.makedirs(to_dir)
    
    args_fn = os.path.join(from_dir, "args.json")
    assert os.path.exists(args_fn), f"{args_fn} does not exist"
    # copy args.json to to_dir
    if not os.path.exists(os.path.join(to_dir, "args.json")):
        shutil.copy(args_fn, to_dir)
    # TODO: copy 
    summary_fn = os.path.join(from_dir, "model_summary.txt")
    if os.path.exists(summary_fn) and (not os.path.exists(os.path.join(to_dir, "model_summary.txt"))):
        shutil.copy(summary_fn, to_dir)

    for i in range(len(epoches)):
        epoch = epoches[i]
        epoch_fn = epoch_fns[i]

        epoch_path = os.path.join(from_dir, epoch_fn)
        epoch_to_path = os.path.join(to_dir, epoch_fn)
        if not os.path.exists(epoch_to_path):
            assert os.path.exists(epoch_path), f"{epoch_path} does not exist"
            # copy epoch-xx.pt to to_dir
            model = torch.load(epoch_path)
            model_small = {"model": model["model"], "epoch": model.get("epoch", epoch)}
            torch.save(model_small, os.path.join(to_dir, epoch_fn))
            logger.success(f"copy {epoch_fn} success")
    logger.success(f"copy {from_dir} to {to_dir}")
    
    return




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, default="nE2024/outputs")
    args = parser.parse_args()
    clear_checkpoint_optimizer_states(dir=args.dir)