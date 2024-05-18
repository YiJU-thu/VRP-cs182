import torch
from torch import nn

from loguru import logger

from utils.functions import sample_many
from utils.beam_search import beam_search, CachedLookup
from utils.tensor_functions import compute_in_batches

from nets.encoder_gat import AttentionEncoder
from nets.encoder_gcn import GCNEncoder
from nets.decoder_gat import AttentionDecoder
from nets.decoder_nAR import NonAutoRegDecoder

import time

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
        return sample_many(self._decoder, input, embed, batch_rep, iter_rep)
    

    def beam_search(self, input, beam_size, compress_mask, max_calc_batch_size):
        # NOTE: for different problems, the input may be different (in old versions)
        fixed = self.precompute_fixed(input)

        def propose_expansions(beam):
            return self.propose_expansions(
                beam, fixed, beam_size, normalize=True, max_calc_batch_size=max_calc_batch_size
            )

        state = self.problem.make_state(
            input, visited_dtype=torch.int64 if compress_mask else torch.uint8
        )

        return beam_search(state, beam_size, propose_expansions)
            
    def precompute_fixed(self, input):
        embed = self._encoder(input)    # embed is a dict, keys specific to the encoder & compatible with the decoder
        # NOTE: _precompute method needs to be implemented in each decoder
        return CachedLookup(self._decoder._precompute(**embed)) # TODO: what is this method doing?
    
    def propose_expansions(self, beam, fixed, expand_size, normalize, max_calc_batch_size):
        # First dim = batch_size * cur_beam_size
        log_p_topk, ind_topk = compute_in_batches(
            lambda b: self._get_log_p_topk(fixed[b.ids], b.state, k=expand_size, normalize=normalize),
            max_calc_batch_size, beam, n=beam.size()
        )

        assert log_p_topk.size(1) == 1, "Can only have single step"
        
        # NOTE: this is the key difference between BS & SGBS
        score_expand = beam.score[:, None] + log_p_topk[:, 0, :]    # This will broadcast, calculate log_p (score) of expansions

        # We flatten the action as we need to filter and this cannot be done in 2d
        flat_action = ind_topk.view(-1)
        flat_score = score_expand.view(-1)
        flat_feas = flat_score > -1e10  # != -math.inf triggers

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