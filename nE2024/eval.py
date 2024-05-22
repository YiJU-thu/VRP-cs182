import math
import torch
import os
import argparse
import numpy as np
import itertools
from tqdm import tqdm
from utils import load_model, move_to
from utils.data_utils import save_dataset
from torch.utils.data import DataLoader
import time
from datetime import timedelta
from utils.functions import parse_softmax_temperature
mp = torch.multiprocessing.get_context('spawn')

from options import get_eval_options



def eval_dataset_mp(args):
    (dataset_path, width, softmax_temp, opts, i, num_processes) = args

    model, _ = load_model(opts.model)
    val_size = opts.val_size // num_processes
    dataset = model.problem.make_dataset(filename=dataset_path, num_samples=val_size, offset=opts.offset + val_size * i)
    device = torch.device("cuda:{}".format(i))

    return _eval_dataset(model, dataset, width, softmax_temp, opts, device)


def eval_dataset(dataset_path, width, softmax_temp, opts, dataset=None):
    # Even with multiprocessing, we load the model here since it contains the name where to write results
    model, _ = load_model(opts.model)
    
    rescale_dist = model.rescale_dist 
    non_Euc = model.non_Euc 
    # assert model.problem.NAME == "tsp", "Only implemented for TSP"

    use_cuda = torch.cuda.is_available() and not opts.no_cuda
    if opts.multiprocessing:
        raise NotImplementedError("Multiprocessing not implemented (tested) for eval")
        assert use_cuda, "Can only do multiprocessing with cuda"
        num_processes = torch.cuda.device_count()
        assert opts.val_size % num_processes == 0

        with mp.Pool(num_processes) as pool:
            results = list(itertools.chain.from_iterable(pool.map(
                eval_dataset_mp,
                [(dataset_path, width, softmax_temp, opts, i, num_processes) for i in range(num_processes)]
            )))

    else:
        device = torch.device("cuda:0" if use_cuda else "cpu")
        dataset = model.problem.make_dataset(filename=dataset_path, num_samples=opts.val_size, offset=opts.offset,
                                             non_Euc=non_Euc, rescale=rescale_dist, dataset=dataset)
        results = _eval_dataset(model, dataset, width, softmax_temp, opts, device)

    # This is parallelism, even if we use multiprocessing (we report as if we did not use multiprocessing, e.g. 1 GPU)
    parallelism = opts.eval_batch_size

    costs, tours, durations = zip(*results)  # Not really costs since they should be negative

    print("Average cost: {} +- {}".format(np.mean(costs), 2 * np.std(costs) / np.sqrt(len(costs))))
    print("Average serial duration: {} +- {}".format(
        np.mean(durations), 2 * np.std(durations) / np.sqrt(len(durations))))
    print("Average parallel duration: {}".format(np.mean(durations) / parallelism))
    print("Calculated total duration: {}".format(timedelta(seconds=int(np.sum(durations) / parallelism))))


    if True: # skip save results
        return costs, tours, durations
    dataset_basename, ext = os.path.splitext(os.path.split(dataset_path)[-1])
    model_name = "_".join(os.path.normpath(os.path.splitext(opts.model)[0]).split(os.sep)[-2:])
    if opts.o is None:
        results_dir = os.path.join(opts.results_dir, model.problem.NAME, dataset_basename)
        os.makedirs(results_dir, exist_ok=True)

        out_file = os.path.join(results_dir, "{}-{}-{}{}-t{}-{}-{}{}".format(
            dataset_basename, model_name,
            opts.decode_strategy,
            width if opts.decode_strategy != 'greedy' else '',
            softmax_temp, opts.offset, opts.offset + len(costs), ext
        ))
    else:
        out_file = opts.f

    assert opts.f or not os.path.isfile(
        out_file), "File already exists! Try running with -f option to overwrite."

    save_dataset((results, parallelism), out_file)

    return costs, tours, durations


def _eval_dataset(model, dataset, width, softmax_temp, opts, device):

    model.to(device)
    model.eval()

    model.set_decode_type(
        "greedy" if opts.decode_strategy in ('bs', 'greedy', 'sgbs', 'shpp') else "sampling",
        temp=softmax_temp)

    dataloader = DataLoader(dataset, batch_size=opts.eval_batch_size)

    results = []
    for batch in tqdm(dataloader, disable=opts.no_progress_bar):
        
        if not model.rescale_dist:
            batch["scale_factors"] = None
        
        batch = move_to(batch, device)

        if opts.EAS != 0:                
            ##### To Do: Add specific path of training dataset for EAS ######
            print('####### Do EAS ######')
            start = time.time()
            if opts.EAS == 1:
                model._encoder = model.eas_encoder(batch, model.problem.NAME, eval_opts = opts)
            elif opts.EAS == 2:
                if model.decoder_name == 'nAR':
                    raise NotImplementedError("EAS Decoder not implemented for NAR")
                model._decoder._get_log_p = model.eas_decoder(batch, model.problem.NAME, eval_opts = opts)
            else:
                raise NotImplementedError("EAS not implemented for EAS = ", opts.EAS)
            EAS_duration = time.time() - start

            #Save the model in .pt file
            # torch.save(model, 'eas_model.pt')

            print('EAS duration: ', EAS_duration)
            print('####### EAS Done ######')

        start = time.time()
        with torch.no_grad():
            if opts.decode_strategy in ('sample', 'greedy'):
                if opts.decode_strategy == 'greedy':
                    assert width == 0, "Do not set width when using greedy"
                    assert opts.eval_batch_size <= opts.max_calc_batch_size, \
                        "eval_batch_size should be smaller than calc batch size"
                    batch_rep = 1
                    iter_rep = 1


                elif width * opts.eval_batch_size > opts.max_calc_batch_size:
                    assert opts.eval_batch_size == 1
                    assert width % opts.max_calc_batch_size == 0
                    batch_rep = opts.max_calc_batch_size
                    iter_rep = width // opts.max_calc_batch_size
                else:
                    batch_rep = width
                    iter_rep = 1
                assert batch_rep > 0
                # This returns (batch_size, iter_rep shape)
                sequences, costs = model.sample_many(batch, batch_rep=batch_rep, iter_rep=iter_rep)

#                 batch_size = len(costs)
#                 ids = torch.arange(batch_size, dtype=torch.int64, device=costs.device)

            elif opts.decode_strategy == 'bs':

                sequences, costs = model.beam_search(
                    batch, beam_size=width,
                    compress_mask=opts.compress_mask,
                    max_calc_batch_size=opts.max_calc_batch_size
                )

            elif opts.decode_strategy == 'sgbs':
                # raise NotImplementedError("Sgbs not implemented now")                
                sequences, costs = model.beam_search(
                    batch, beam_size=width,
                    compress_mask=opts.compress_mask,
                    max_calc_batch_size=opts.max_calc_batch_size,
                    sgbs = True,
                    gamma = opts.gamma,
                )

            elif opts.decode_strategy == 'shpp':
                raise NotImplementedError("SHPP not implemented now")
                sequences, costs = model.shpp_refine(
                    batch, ...,
                )


        duration = time.time() - start
        for seq, cost in zip(sequences, costs):
            if model.problem.NAME == "tsp":
                seq = seq.tolist()  # No need to trim as all are same length
            elif model.problem.NAME in ("cvrp", "sdvrp"):
                seq = np.trim_zeros(seq).tolist() + [0]  # Add depot
            elif model.problem.NAME in ("op", "pctsp"):
                seq = np.trim_zeros(seq)  # We have the convention to exclude the depot
            else:
                assert False, "Unkown problem: {}".format(model.problem.NAME)
            # Note VRP only
            results.append((cost, seq, duration))

    return results


def eval(opts):
    for width in opts.widths:
        for dataset_path in opts.datasets:
            eval_dataset(dataset_path, width, opts.softmax_temperature, opts)



if __name__ == "__main__":
    eval(get_eval_options())