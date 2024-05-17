import os, sys
from loguru import logger

curr_dir = os.path.dirname(__file__)
util_dir = os.path.join(curr_dir, '..', 'utils_project')
if util_dir not in sys.path:
    sys.path.append(util_dir)

from clear_optimizer_states import copy_trained_nets
import json


if __name__ == '__main__':
    # arg: default=best, max
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--default_epoch", type=str, default="best", help="best / max")
    parser.add_argument("--from_dir", type=str, default="outputs")
    parser.add_argument("--to_dir", type=str, default="pretrained/neurips")
    args = parser.parse_args()

    copy_json_fn = os.path.join(curr_dir, 'move_pretrained.json')
    with open(copy_json_fn, 'r') as f:
        copy_dict = json.load(f)

    from_dir = os.path.join(curr_dir, args.from_dir)
    to_dir = os.path.join(curr_dir, args.to_dir)

    for from_name, to_name in copy_dict.items():
        # from_name can be either a directory or a file
        # if it is a directory, copy the 'best' (default) / 'last' model
        from_path = os.path.join(from_dir, from_name)
        if not os.path.isdir(from_path):
            logger.info(f"{from_name} does not exists! Skipped!")
            continue

        if isinstance(to_name, list):   # [to_name, epochs to move]
            to_name, epoches = to_name
        else:
            epoches = [args.default_epoch]
        to_path = os.path.join(to_dir, to_name)
        copy_trained_nets(from_path, to_path, epoches=epoches)
        # logger.info(f'Copied {from_path} to {to_path}')
    logger.success('Done')