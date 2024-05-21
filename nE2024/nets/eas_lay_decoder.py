import math
import time
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import inspect
# from nets.decoder_gat import AttentionDecoder
from utils.functions import do_batch_rep


# EAS-Lay parameters
ACTOR_WEIGHT_DECAY = 1e-6
param_lr = 0.0032 # EAS Learning rate
p_runs = 1  # Number of parallel runs per instance, set 1 here
max_iter = 100 # Maximum number of EAS iterations
param_lambda = 1 # Imitation learning loss weight
max_runtime = 1000 # Maximum runtime in seconds

use_cuda = torch.cuda.is_available()

def get_decoder_parameters(decoder):
    params = {}
    init_signature = inspect.signature(decoder.__init__)
    for name, param in init_signature.parameters.items():
        if name != 'self':  # Ignore 'self' parameter
            params[name] = getattr(decoder, name, param.default)
    return params

# ########################################################
class prob_calc_added_layers(nn.Module):
    def __init__(self, base_decoder, embedding_dim):
        super().__init__()
        self.base_decoder = base_decoder

        # Parameters for the new residual layer
        self.new_weight_1 = nn.Parameter(torch.Tensor(embedding_dim, embedding_dim),requires_grad=True)
        self.new_bias_1 = nn.Parameter(torch.Tensor(embedding_dim),requires_grad=True)
        self.new_weight_2 = nn.Parameter(torch.zeros(embedding_dim, embedding_dim),requires_grad=True)
        self.new_bias_2 = nn.Parameter(torch.zeros(embedding_dim),requires_grad=True)
        torch.nn.init.xavier_uniform_(self.new_weight_1)
        torch.nn.init.constant_(self.new_bias_1, 0)
    
    def query_transform(self, query):
        query_eas = torch.matmul(query, self.new_weight_1) + self.new_bias_1
        query_eas = F.relu(query_eas)
        query_eas = torch.matmul(query_eas, self.new_weight_2) + self.new_bias_2
        return query + query_eas
    
    def _get_log_p(self, fixed, state, glimpse=None, normalize=True):        
        graph_context_query = self.base_decoder.project_fixed_context(glimpse)[:, None, :] if glimpse is not None else fixed.context_node_projected

        # Compute query = context node embedding
        query = graph_context_query + \
                self.base_decoder.project_step_context(self.base_decoder._get_parallel_step_context(fixed.node_embeddings, state))
        query = self.query_transform(query)
        # Compute keys and values for the nodes
        glimpse_K, glimpse_V, logit_K = self.base_decoder._get_attention_node_data(fixed, state)

        # Compute the mask
        mask = state.get_mask()

        # Compute logits (unnormalized log_p)
        log_p, glimpse = self.base_decoder._one_to_many_logits(query, glimpse_K, glimpse_V, logit_K, mask)

        if normalize:
            log_p = torch.log_softmax(log_p / self.base_decoder.temp, dim=-1)

        assert not torch.isnan(log_p).any()

        return log_p, mask, glimpse
    
    def _inner(self, input, embeddings, graph_embed=None, force_steps=0):

        outputs = []
        sequences = []

        state = self.base_decoder.problem.make_state(input)

        # Compute keys, values for the glimpse and keys for the logits once as they can be reused in every step
        fixed = self.base_decoder._precompute(embeddings, graph_embed=graph_embed)
        glimpse = embeddings.mean(1)

        batch_size = state.ids.size(0)

        # Perform decoding steps
        i = 0

        while not (self.base_decoder.shrink_size is None and state.all_finished()):

            if self.base_decoder.shrink_size is not None:
                unfinished = torch.nonzero(state.get_finished() == 0)
                if len(unfinished) == 0:
                    break
                unfinished = unfinished[:, 0]
                # Check if we can shrink by at least shrink_size and if this leaves at least 16
                # (otherwise batch norm will not work well and it is inefficient anyway)
                if 16 <= len(unfinished) <= state.ids.size(0) - self.base_decoder.shrink_size:
                    # Filter states
                    state = state[unfinished]
                    fixed = fixed[unfinished]

            assert glimpse.shape == (batch_size, self.base_decoder.embedding_dim), "glimpse shape is {}".format(glimpse.shape)
            if not self.base_decoder.update_context_node:
                glimpse = None
            log_p, mask, glimpse = self._get_log_p(fixed, state, glimpse)
            
            if i >= force_steps:
                # Select the indices of the next nodes in the sequences, result (batch_size) long
                selected = self.base_decoder._select_node(log_p.exp()[:, 0, :], mask[:, 0, :])  # Squeeze out steps dimension
            else:
                selected = torch.tensor([i]*batch_size, device=log_p.device)
                # NOTE: SHPP: 0 -> 1 -> xxx
                # so when use SHPP mode to solve SHPP, let terminal=0, start=1

            state = state.update(selected)

            # Now make log_p, selected desired output size by 'unshrinking'
            if self.base_decoder.shrink_size is not None and state.ids.size(0) < batch_size:
                log_p_, selected_ = log_p, selected
                log_p = log_p_.new_zeros(batch_size, *log_p_.size()[1:])
                selected = selected_.new_zeros(batch_size)

                log_p[state.ids[:, 0]] = log_p_
                selected[state.ids[:, 0]] = selected_

            # Collect output of step
            outputs.append(log_p[:, 0, :])
            sequences.append(selected)

            i += 1

        # Collected lists, return Tensor
        return torch.stack(outputs, 1), torch.stack(sequences, 1)

    def forward(self, input, embeddings, graph_embed=None, force_steps=0, return_pi=False):
        # Forward pass through the base decoder
        _log_p, pi = self._inner(input, embeddings, graph_embed=graph_embed, force_steps=force_steps)

        for i in range(force_steps):
            assert pi[:, i].eq(i).all(), "Forced output incorrect"
        
        cost, mask = self.base_decoder.problem.get_costs(input, pi)
        # Log likelyhood is calculated within the model since returning it per action does not work well with
        # DataParallel since sequences can be of different lengths
        ll = self.base_decoder._calc_log_likelihood(_log_p, pi, mask, force_steps=force_steps)
        if return_pi:
            return cost, ll, pi

        return cost, ll


def replace_decoder(grouped_actor, state, decoder_params):
    """Function to add layers to pretrained model while retaining weights from other layers."""
    grouped_actor = prob_calc_added_layers(grouped_actor, decoder_params['embedding_dim'])
    # grouped_actor.load_state_dict(state_dict=state, strict=False)
    return grouped_actor


#######################################################################################################################
# EAS Training process
#######################################################################################################################

def run_eas_lay_decoder(encoder, grouped_actor, instance_data, problem_name, eval_opts):
    # Save the original state_dict of the decoder (e.p. the parameters of the decoder)
    original_decoder_state_dict = grouped_actor.state_dict()
    decoder_params = get_decoder_parameters(grouped_actor)


    # Load the instances
    ###############################################
    with torch.no_grad():
    # Didn't do Augment here because it's not necessary for the Non-eculidean case
        if use_cuda:
            grouped_actor_modified = replace_decoder(grouped_actor, original_decoder_state_dict,
                                                     decoder_params).cuda()
        else:
            print("No GPU available")
            grouped_actor_modified = replace_decoder(grouped_actor, original_decoder_state_dict,
                                                     decoder_params)

    # Only update the weights of the added layer during training
    optimizer = optim.Adam(
        [grouped_actor_modified.new_weight_1,
        grouped_actor_modified.new_bias_1,
        grouped_actor_modified.new_weight_2,
        grouped_actor_modified.new_bias_2], lr=param_lr,
        weight_decay=ACTOR_WEIGHT_DECAY)


    # Start the search
    ###############################################
    t_start = time.time()
    for iter in range(max_iter):
        embed = encoder(instance_data)
        batch_rep=1
        iter_rep=1
        input, embed = do_batch_rep((instance_data, embed), batch_rep)

        costs = []
        _log_p_s = []
        for i in range(iter_rep):
            cost, _log_p, pi = grouped_actor_modified(input, return_pi=True, **embed)

            costs.append(cost.view(batch_rep, -1).t())
            _log_p_s.append(_log_p.view(batch_rep, -1).t())

        costs = torch.cat(costs, 1)
        _log_p_s = torch.cat(_log_p_s, 1)

        loss_1 = costs.mean()
        loss_2 = -_log_p_s.mean()
        loss = loss_1 + param_lambda*loss_2

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if time.time() - t_start > max_runtime:
            break
    return grouped_actor_modified._get_log_p