import torch
from torch import nn

from loguru import logger

from utils.functions import sample_many, gpu_memory_usage
from utils.beam_search import beam_search, CachedLookup
from utils.tensor_functions import compute_in_batches

from nets.encoder_gat import AttentionEncoder
from nets.encoder_gcn import GCNEncoder
from nets.decoder_gat import AttentionDecoder
from nets.decoder_nAR import NonAutoRegDecoder
import time
from copy import deepcopy

from nets.eas_lay_decoder import run_eas_lay_decoder
from nets.eas_lay_encoder import run_eas_lay_encoder
# from options import get_options, get_eval_options

import os, sys
curr_dir = os.path.dirname(os.path.abspath(__file__))
utils_proj_dir = os.path.join(curr_dir, "../../utils_project")
if utils_proj_dir not in sys.path: 
    sys.path.append(utils_proj_dir)
from utils_vrp import normalize_graph, get_tour_len_torch


class VRPModel(nn.Module):

    encoders = {
        "gat": AttentionEncoder,
        "gcn": GCNEncoder,
    }

    decoders = {
        "gat": AttentionDecoder,
        "nAR": NonAutoRegDecoder,
    }

    def __init__(self, encoder_name, decoder_name, encoder_kws, decoder_kws):
        super(VRPModel, self).__init__()

        self.encoder_name = encoder_name
        self.decoder_name = decoder_name

        # avoid en/decoders being called directly outside the model
        self._encoder = self.encoders[encoder_name](**encoder_kws)
        self._decoder = self.decoders[decoder_name](**decoder_kws)
        # NOTE: do not change attr names if the model params are to be loaded from a file
        # so in old versions, as I use self.encoder/decoder, I still need these names to have the params correctly loaded
        self.encoder, self.decoder = self._encoder, self._decoder



        self.time_count = {
            "encoder_forward": 0, "decoder_forward": 0, "model_update": 0, "data_gen": 0, "baseline_eval": 0,
        }


        self.rescale_dist = self._encoder.rescale_dist
        self.non_Euc = self._encoder.non_Euc
        self.problem = self._encoder.problem


    
    def forward(self, input, ref_pi=None, **kws):
        """
        :param input: (batch_size, graph_size, node_dim) input node features or dictionary with multiple tensors
        :param ref_pi: ref_pi: reference sequence to calculate the log likelihood
        :return:
        FIXME: in the notes, make it clear what **kw probably are
        """

        t0 = time.perf_counter()
        embed = self._encoder(input)    # embed is a dict, keys specific to the encoder & compatible with the decoder
        embed.update(kws)

        t1 = time.perf_counter()
        res = self._decoder(input, ref_pi=ref_pi, **embed)
        
        # gpu_memory_usage(msg="Forward", on=True)

        # NOTE: old version, no longer needed
        # if self.decoder_name == "gat":
        #     embeddings, graph_embed = self._encoder(input)
        #     t1 = time.perf_counter()
        #     res = self._decoder(input, embeddings, graph_embed=graph_embed, **kws)
            
        # elif self.decoder_name == "nAR":
        #     heatmap = self._encoder(input)
        #     t1 = time.perf_counter()
        #     res = self._decoder(input, heatmap, **kws)
        
        t2 = time.perf_counter()
        self.update_time_count(encoder_forward=t1-t0, decoder_forward=t2-t1)
        
        return res

    
    def get_loss(self, input, ref_pi=None, loss_type="rl", loss_params=None, **kws):
        """
        :param input: (batch_size, graph_size, node_dim) input node features or dictionary with multiple tensors
        :param ref_pi: reference sequence to calculate the log likelihood
        :param loss_params: dict
        :return:
        """

        if loss_type == "ul":   # unsupervised learning
            loss, loss_misc = self.get_ul_loss(input, c1=loss_params["c1"])
        else:
            raise NotImplementedError(f"Loss type {loss_type} not implemented")
        return loss, loss_misc
            


    def get_ul_loss(self, input, c1=10):
        
        dist_mat = input["distance"]
        heatmap = self._encoder(input)["heatmap"]   # (I, N, N), log_p
        prob_mat = torch.softmax(heatmap, dim=-1) # each row sum to 1, now it becomes a probability matrix
        
        # make diagonal elements 0, and normalize the rows
        prob_mat = prob_mat * (1 - torch.eye(prob_mat.size(1), device=prob_mat.device))
        prob_mat = prob_mat / prob_mat.sum(dim=-1, keepdim=True)

        tsp_loss = torch.sum(prob_mat * dist_mat, dim=(1,2))
        col_sum_one_loss = torch.sum((1-torch.sum(prob_mat, dim=1))**2, dim=-1) # push the sum of each column to 1
        loss = tsp_loss + c1 * col_sum_one_loss
        assert loss.shape == (input["distance"].shape[0],), f"loss shape: {loss.shape}"

        return torch.mean(loss), {"tsp_loss": torch.mean(tsp_loss), "col_sum_one_loss": torch.mean(col_sum_one_loss)}
        



    def set_decode_type(self, decode_type, temp=None):
        self._decoder.set_decode_type(decode_type, temp)
    
    
    def sample_many(self, input, batch_rep=1, iter_rep=1):
        embed = self._encoder(input)    # embed is a dict, keys specific to the encoder & compatible with the decoder
        res = sample_many(self._decoder, input, embed, batch_rep, iter_rep)
        gpu_memory_usage(msg="Sample many", on=True)
        return res
    

    def beam_search(self, input, beam_size, compress_mask, max_calc_batch_size, sgbs=False, gamma = 0):
        # NOTE: for different problems, the input may be different (in old versions)
        fixed = self.precompute_fixed(input)

        def propose_expansions(beam, sgbs = sgbs):
            return self.propose_expansions(
                beam, fixed, beam_size*(1-sgbs)+gamma*sgbs, normalize=True, max_calc_batch_size=max_calc_batch_size, sgbs = sgbs, beam_size = beam_size
            )
        state = self.problem.make_state(
            input, visited_dtype=torch.int64 if compress_mask else torch.uint8
        )

        res = beam_search(state, beam_size, propose_expansions)
        gpu_memory_usage(msg="Beam search", on=True)
        return res
    

    def shpp_refine(self, input, init_method="rand_insert"):
        
        
        n_splits = ...
        time_limit = ...

        t0 = time.perf_counter()

        pi_0, cost_0 = ...  # get initial solution, shape (batch_size, graph_size)
        logger.debug("Before refinement: {:.3f}+-{:.3f}".format(cost_0.mean(), cost_0.std()))
        
        best_cost = cost_0
        i = 1
        while time.perf_counter() - t0 < time_limit:
            pi_1, cost_1 = self._shpp_refine(input, pi_0, ...)
            
            
            # make cost_1 be the minimum of cost_0 and cost_1, and update pi_0
            mask = cost_1 < cost_0
            pi_0[mask], cost_0[mask] = pi_1[mask], cost_1[mask]
            cost_diff = cost_0 - best_cost
            logger.debug("After refinement {}: {:.3f} ({:.2%})".format(i, cost_diff.mean(), mask.mean()))

            i += 1
            best_cost = cost_0

        return pi_1, cost_1
    

    def _shpp_refine(self, input, pi_0, n_splits=2, idx_0=0):
        # to maximize the efficiency, we require every segment be the same length

        def _reorder_input(input, pi_0_expand):
            input_reordered = {}
            for k, v in input.items():
                if v is not None:
                    if k in ["distance", "rel_distance"]:
                        input_reordered[k] = v[...] # BUG
                    else:
                        input_reordered[k] = v[...] # BUG
                else:
                    input_reordered[k] = None
            return input_reordered

        def _get_sub_input(input, start, end):
            sub_input = {}
            for k, v in input.items():
                if v is not None:
                    if k in ["distance", "rel_distance"]:
                        sub_input[k] = input[k][:, start:end, start:end]
                    else:
                        sub_input[k] = input[k][:, start:end]
                else:
                    sub_input[k] = None
            return sub_input

        def _merge_splits(input_splits: list):
            keys = input_splits[0].keys()
            merged = {}
            for k in keys:
                if input_splits[0][k] is not None:
                    merged[k] = torch.cat([x[k] for x in input_splits], dim=0)   # (batch_size * n_splits, ...)
                else:
                    merged[k] = None
            return merged

        def _merge_sols(split_pis, idx_splits):
            raise NotImplementedError("Not implemented yet")


        if "coords" in input:
            batch_size, n = input["coords"].shape[:2]
        elif "distance" in input:
            batch_size, n = input["distance"].shape[:2]


        # equivalent to: shift (to left) idx_0 columns, and append the first column to the end
        pi_0_expand = torch.cat([pi_0[:,idx_0:], pi_0[:,:idx_0+1]], dim=1)
        input_expand = _reorder_input(input, pi_0_expand) # reorder the data accordingly

        len_split = np.ceil(n / n_splits).astype(int)   # each split has (approx) len_split nodes

        input_splits = [None] * n_splits
        idx_splits = [None] * n_splits
        start = 0

        for i in range(n_splits):   # FIXME: may improve later
            l = len_split + 1   # max(len_split + 1 + np.random.randint(-3,3), 1) # add some randomness to the length of each split
            end = min(start + l, n+1)
            # if i == n_splits - 1:
            #     end = nodes_tour0_expand.shape[1]
            
            # coords = nodes_tour0_expand[:, start:end, :]    # (I, (approx) len_split, 2)
            sub_input = _get_sub_input(input_expand, start, end)
            sub_input = normalize_graph(sub_input, rescale=False) # FIXME
            
            input_splits[i] = sub_input
            idx_splits[i] = pi_0_expand[:, start:end]
            start += l-1
        
        merged_mini_batch = _merge_splits(input_splits)  # (batch_size * n_splits, len_split+1, ...)
        merged_cost, _, merged_pi  = self.forward(merged_mini_batch, force_steps=2, return_pi=True) # FIXME: later, can use ref_pi
        assert merged_pi.shape == (batch_size * n_splits, len_split+1), f"merged_pi.shape: {merged_pi.shape}"
        # reshape to (batch_size, n_splits, len_split+1)
        split_pis = merged_pi.view(batch_size, n_splits, len_split+1).roll(-1, dims=2)  # make the first to be the last
        refined_pi = _merge_sols(split_pis, idx_splits)  # (batch_size, n)
        refined_cost = get_tour_len_torch(input, refined_pi)

        return refined_pi, refined_cost

        tour_to_concat = [None] * n_splits
        for i in range(n_splits):
            s = shpp_splits[i]
            tour_to_concat[i] = idx_splits[i][range(s.shape[0]), s.T].T[:,:-1]
        tours1 = np.concatenate(tour_to_concat, axis=1)
        assert tours1.shape == tours0.shape, "tours1.shape={}, tours0.shape={}".format(tours1.shape, tours0.shape)
        obj1 = tour_len_euc_2d(data, tours1)

        res1 = {"obj": obj1, "tours": tours1, "time": res0["time"] + durations}
        
        if not mute:
            print("After refinement: {:.3f}+-{:.3f}".format(obj1.mean(), obj1.std()))
            pass



    def precompute_fixed(self, input):
        embed = self._encoder(input)    # embed is a dict, keys specific to the encoder & compatible with the decoder
        # NOTE: _precompute method needs to be implemented in each decoder
        return CachedLookup(self._decoder._precompute(**embed)) # TODO: what is this method doing?
    
    def propose_expansions(self, beam, fixed, expand_size, normalize, max_calc_batch_size, sgbs, beam_size):
        # First dim = batch_size * cur_beam_size
        log_p_topk, ind_topk = compute_in_batches(
            lambda b: self._get_log_p_topk(fixed[b.ids], b.state, k=expand_size, normalize=normalize),
            max_calc_batch_size, beam, n=beam.size()
        )
        assert log_p_topk.size(1) == 1, "Can only have single step"

        # NOTE: this is the key difference between BS & SGBS
        if not sgbs:    
            score_expand = beam.score[:, None] + log_p_topk[:, 0, :]    # This will broadcast, calculate log_p (score) of expansions

        else:
            rollout_cost_topk = compute_in_batches(
            lambda b: self._get_rollout_cost_topk(fixed[b.ids], b.state,\
                                                  log_p_topk,\
                                                  ind_topk,\
                                                  k=expand_size, normalize=normalize),
            max_calc_batch_size, beam, n=beam.size()
            )
            score_expand = - rollout_cost_topk[:, 0, :]

        # We flatten the action as we need to filter and this cannot be done in 2d
        flat_action = ind_topk.view(-1)
        flat_score = score_expand.view(-1)


        flat_feas = log_p_topk[:, 0, :].view(-1) > -1e10  # != -math.inf triggers

        # Parent is row idx of ind_topk, can be found by enumerating elements and dividing by number of columns
        flat_parent = torch.arange(flat_action.size(-1), out=flat_action.new()) // ind_topk.size(-1)

        # Filter infeasible
        feas_ind_2d = torch.nonzero(flat_feas)

        if len(feas_ind_2d) == 0:
            # Too bad, no feasible expansions at all :(
            return None, None, None

        feas_ind = feas_ind_2d[:, 0]
        return flat_parent[feas_ind], flat_action[feas_ind], flat_score[feas_ind]

    def _get_log_p_topk(self, fixed, state, k, normalize):
        # NOTE: _get_log_p method needs to be implemented in each decoder
        log_p, mask, glimpse = self._decoder._get_log_p(fixed, state, normalize=normalize)

        # Return topk
        if k is not None and k < log_p.size(-1):
            return log_p.topk(k, -1)    # Tensor.topk(k, dim) returns (values, indices)

        # Return all, note different from torch.topk this does not give error if less than k elements along dim
        return (
            log_p,
            torch.arange(log_p.size(-1), device=log_p.device, dtype=torch.int64).repeat(log_p.size(0), 1)[:, None, :]
        )

    def _get_rollout_cost_topk(self, fixed, state, log_p_topk, ind_topk, k, normalize):
        # Save the original decode_type
        original_decode_type = self._decoder.decode_type
        self._decoder.decode_type = "greedy"

        rollout_cost_topk = torch.zeros_like(log_p_topk)
        for i in range(k):
            rollout_cost = self.roll_out_simulation(fixed, state, ind_topk[:, 0, i].squeeze(), normalize)
            rollout_cost_topk[:, :, i] = rollout_cost

        # Restore the original decode_type
        self._decoder.decode_type = original_decode_type

        return rollout_cost_topk
    
    def roll_out_simulation(self, fixed, state, next_node, normalize):
        _state = deepcopy(state)
        _state = _state.update(next_node)
        # Start roll-out from the next node
        while not _state.all_finished():
            log_p, mask, glimpse = self._decoder._get_log_p(fixed, _state, normalize=normalize)
            selected = self._decoder._select_node(log_p.exp()[:, 0, :], mask[:, 0, :])
            _state = _state.update(selected)
        return _state.get_final_cost()

    def update_time_count(self, **kw):
        for key in kw:
            self.time_count[key] += kw[key]
    
    @property
    def time_stats(self):
        total_time = sum(self.time_count.values())
        if total_time == 0:

           info = {f'T-{key}': 0 for key in self.time_count}
        else:
            info = {f'T-{key}': self.time_count[key] / total_time for key in self.time_count}
        info["total_time"] = total_time / 3600 # in hours
        return info
    
    def eas_encoder(self, input, problem_name, eval_opts, max_runtime=1000):
        # raise NotImplementedError("EAS not implemented for encoder")
        return run_eas_lay_encoder(self._encoder, self._decoder, input, self.encoder_name, problem_name, eval_opts, max_runtime=max_runtime)

    def eas_decoder(self, input, problem_name, eval_opts, max_runtime=1000):
        # raise NotImplementedError("EAS not implemented for decoder")
        return run_eas_lay_decoder(self._encoder, self._decoder, input, problem_name, eval_opts, max_runtime=max_runtime)