import math
import time
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
# Import the necessary functions from options.py
from options import get_options, get_eval_options

#### Currently used, should be in other files #############
# Hyper-Parameters
EMBEDDING_DIM = 128
KEY_DIM = 16  # Length of q, k, v of EACH attention head
HEAD_NUM = 8
ENCODER_LAYER_NUM = 6
FF_HIDDEN_DIM = 512
LOGIT_CLIPPING = 10  # (C in the paper)

ACTOR_WEIGHT_DECAY = 1e-6

# EAS-Lay parameters
param_lr = 0.0032
p_runs = 1  # Number of parallel runs per instance, set 1 here
max_iter = 200 # Maximum number of EAS iterations
param_lambda = 0.012
max_runtime = 1000 # Maximum runtime in seconds

use_cuda = torch.cuda.is_available()

FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
BoolTensor = torch.cuda.BoolTensor if use_cuda else torch.BoolTensor
Tensor = FloatTensor

def reshape_by_heads_TSP(qkv, head_num):
    # q.shape = (batch, C, head_num*key_dim)

    batch_s = qkv.size(0)
    C = qkv.size(1)

    q_reshaped = qkv.reshape(batch_s, C, head_num, -1)
    # shape = (batch, C, head_num, key_dim)

    q_transposed = q_reshaped.transpose(1, 2)
    # shape = (batch, head_num, C, key_dim)

    return q_transposed

def multi_head_attention_TSP(q, k, v, ninf_mask=None, group_ninf_mask=None):
    # q shape = (batch_s, head_num, n, key_dim)   : n can be either 1 or TSP_SIZE
    # k,v shape = (batch_s, head_num, TSP_SIZE, key_dim)
    # ninf_mask.shape = (batch_s, TSP_SIZE)
    # group_ninf_mask.shape = (batch_s, group, TSP_SIZE)

    batch_s = q.size(0)
    head_num = q.size(1)
    n = q.size(2)
    key_dim = q.size(3)
    problem_s = k.size(2)

    score = torch.matmul(q, k.transpose(2, 3))
    # shape = (batch_s, head_num, n, TSP_SIZE)

    score_scaled = score / np.sqrt(key_dim)
    if ninf_mask is not None:
        score_scaled = score_scaled + ninf_mask[:, None, None, :].expand(batch_s, head_num, n, problem_s)
    if group_ninf_mask is not None:
        score_scaled = score_scaled + group_ninf_mask[:, None, :, :].expand(batch_s, head_num, n, problem_s)

    weights = nn.Softmax(dim=3)(score_scaled)
    # shape = (batch_s, head_num, n, TSP_SIZE)

    out = torch.matmul(weights, v)
    # shape = (batch_s, head_num, n, key_dim)

    out_transposed = out.transpose(1, 2)
    # shape = (batch_s, n, head_num, key_dim)

    out_concat = out_transposed.reshape(batch_s, n, head_num * key_dim)
    # shape = (batch_s, n, head_num*key_dim)

    return out_concat

class GROUP_STATE:

    def __init__(self, group_size, data, problem_size):
        # data.shape = (batch, group, 2)
        self.batch_s = data.size(0)
        self.group_s = group_size
        self.data = data
        self.problem_size = problem_size

        # History
        ####################################
        self.selected_count = 0
        self.current_node = None
        # shape = (batch, group)
        self.selected_node_list = LongTensor(np.zeros((self.batch_s, group_size, 0)))
        # shape = (batch, group, selected_count)

        # Status
        ####################################
        self.ninf_mask = Tensor(np.zeros((self.batch_s, group_size, problem_size)))
        # shape = (batch, group, TSP_SIZE)

    def move_to(self, selected_idx_mat):
        # selected_idx_mat.shape = (batch, group)

        # History
        ####################################
        self.selected_count += 1
        self.current_node = selected_idx_mat
        self.selected_node_list = torch.cat((self.selected_node_list, selected_idx_mat[:, :, None]), dim=2)

        # Status
        ####################################
        batch_idx_mat = torch.arange(self.batch_s)[:, None].expand(self.batch_s, self.group_s)
        group_idx_mat = torch.arange(self.group_s)[None, :].expand(self.batch_s, self.group_s)
        self.ninf_mask[batch_idx_mat, group_idx_mat, selected_idx_mat] = -np.inf

class GROUP_ENVIRONMENT:

    def __init__(self, data, problem_size, round_distances = False):
        # seq.shape = (batch, TSP_SIZE, 2)

        self.data = data
        self.batch_s = data.size(0)
        self.group_s = None
        self.group_state = None
        self.problem_size = problem_size
        self.round_distances = round_distances

    def reset(self, group_size):
        self.group_s = group_size
        self.group_state = GROUP_STATE(group_size=group_size, data=self.data, problem_size=self.problem_size)
        reward = None
        done = False
        return self.group_state, reward, done

    def step(self, selected_idx_mat):
        # selected_idx_mat.shape = (batch, group)

        # move state
        self.group_state.move_to(selected_idx_mat)

        # returning values
        done = (self.group_state.selected_count == self.problem_size)
        if done:
            reward = -self._get_group_travel_distance()  # note the minus sign!
        else:
            reward = None
        return self.group_state, reward, done

    def _get_group_travel_distance(self):
        gathering_index = self.group_state.selected_node_list.unsqueeze(3).expand(self.batch_s, -1, self.problem_size,
                                                                                  2)
        # shape = (batch, group, TSP_SIZE, 2)
        seq_expanded = self.data[:, None, :, :].expand(self.batch_s, self.group_s, self.problem_size, 2)

        ordered_seq = seq_expanded.gather(dim=2, index=gathering_index)
        # shape = (batch, group, TSP_SIZE, 2)

        rolled_seq = ordered_seq.roll(dims=2, shifts=-1)
        segment_lengths = ((ordered_seq - rolled_seq) ** 2).sum(3).sqrt()
        # size = (batch, group, TSP_SIZE)

        group_travel_distances = segment_lengths.sum(2)
        # size = (batch, group)
        return group_travel_distances

def get_episode_data(data, episode, batch_size, problem_size):
    node_data = Tensor(data[0][episode:episode + batch_size])

    return node_data, None

def augment_xy_data_by_8_fold(xy_data):
    # xy_data.shape = (batch_s, problem, 2)

    x = xy_data[:, :, [0]]
    y = xy_data[:, :, [1]]
    # x,y shape = (batch, problem, 1)
    dat1 = torch.cat((x, y), dim=2)
    dat2 = torch.cat((1 - x, y), dim=2)
    dat3 = torch.cat((x, 1 - y), dim=2)
    dat4 = torch.cat((1 - x, 1 - y), dim=2)
    dat5 = torch.cat((y, x), dim=2)
    dat6 = torch.cat((1 - y, x), dim=2)
    dat7 = torch.cat((y, 1 - x), dim=2)
    dat8 = torch.cat((1 - y, 1 - x), dim=2)

    data_augmented = torch.cat((dat1, dat2, dat3, dat4, dat5, dat6, dat7, dat8), dim=0)
    # shape = (8*batch, problem, 2)

    return data_augmented

def augment_and_repeat_episode_data(episode_data, problem_size, nb_runs, aug_s):
    node_data = episode_data[0]

    batch_size = node_data.shape[0]

    node_xy = node_data

    if nb_runs > 1:
        assert batch_size == 1
        node_xy = node_xy.repeat(nb_runs, 1, 1)

    if aug_s > 1:
        assert aug_s == 8
        # 8 fold Augmented
        # aug_depot_xy.shape = (8*batch, 1, 2)
        node_xy = augment_xy_data_by_8_fold(node_xy)
        # aug_node_xy.shape = (8*batch, problem, 2)
        # aug_node_demand.shape = (8*batch, problem, 2)

    return (node_xy)

# ########################################################

AUG_S = 8

class prob_calc_added_layers_TSP(nn.Module):
    """New nn.Module with added layers for the TSP.

    same as source.MODEL__Actor.grouped.actors.Next_Node_Probability_Calculator_for_group with added layers.
    """

    def __init__(self, batch_s):
        super().__init__()

        self.Wq_graph = nn.Linear(EMBEDDING_DIM, HEAD_NUM * KEY_DIM, bias=False)
        self.Wq_first = nn.Linear(EMBEDDING_DIM, HEAD_NUM * KEY_DIM, bias=False)
        self.Wq_last = nn.Linear(EMBEDDING_DIM, HEAD_NUM * KEY_DIM, bias=False)
        self.Wk = nn.Linear(EMBEDDING_DIM, HEAD_NUM * KEY_DIM, bias=False)
        self.Wv = nn.Linear(EMBEDDING_DIM, HEAD_NUM * KEY_DIM, bias=False)

        self.new = nn.Parameter(torch.zeros((batch_s, HEAD_NUM * KEY_DIM, HEAD_NUM * KEY_DIM), requires_grad=True))
        self.new_bias = nn.Parameter(torch.zeros((batch_s, 1, HEAD_NUM * KEY_DIM), requires_grad=True))
        self.new_2 = nn.Parameter(torch.zeros((batch_s, HEAD_NUM * KEY_DIM, HEAD_NUM * KEY_DIM), requires_grad=True))
        self.new_bias_2 = nn.Parameter(torch.zeros((batch_s, 1, HEAD_NUM * KEY_DIM), requires_grad=True))

        torch.nn.init.xavier_uniform(self.new)
        torch.nn.init.xavier_uniform(self.new_bias)

        self.multi_head_combine = nn.Linear(HEAD_NUM * KEY_DIM, EMBEDDING_DIM)

        self.q_graph = None  # saved q1, for multi-head attention
        self.q_first = None  # saved q2, for multi-head attention
        self.k = None  # saved key, for multi-head attention
        self.v = None  # saved value, for multi-head_attention
        self.single_head_key = None  # saved, for single-head attention

    def reset(self, encoded_graph, encoded_nodes):
        # encoded_nodes.shape = (batch_s, TSP_SIZE, EMBEDDING_DIM)

        self.q_graph = reshape_by_heads_TSP(self.Wq_graph(encoded_graph), head_num=HEAD_NUM)
        # shape = (batch_s, HEAD_NUM, 1, KEY_DIM)
        self.q_first = None
        # shape = (batch_s, HEAD_NUM, group, KEY_DIM)
        self.k = reshape_by_heads_TSP(self.Wk(encoded_nodes), head_num=HEAD_NUM)
        self.v = reshape_by_heads_TSP(self.Wv(encoded_nodes), head_num=HEAD_NUM)
        # shape = (batch_s, HEAD_NUM, TSP_SIZE, KEY_DIM)
        self.single_head_key = encoded_nodes.transpose(1, 2)
        # shape = (batch_s, EMBEDDING_DIM, TSP_SIZE)
        # self.group_ninf_mask = group_ninf_mask
        # shape = (batch_s, group, TSP_SIZE)

    def forward(self, encoded_LAST_NODE, group_ninf_mask):
        # encoded_LAST_NODE.shape = (batch_s, group, EMBEDDING_DIM)

        with torch.no_grad():
            if self.q_first is None:
                self.q_first = reshape_by_heads_TSP(self.Wq_first(encoded_LAST_NODE), head_num=HEAD_NUM)
            # shape = (batch_s, HEAD_NUM, group, KEY_DIM)

            #  Multi-Head Attention
            #######################################################
            q_last = reshape_by_heads_TSP(self.Wq_last(encoded_LAST_NODE), head_num=HEAD_NUM)
            # shape = (batch_s, HEAD_NUM, group, KEY_DIM)

            q = self.q_graph + self.q_first + q_last
            # shape = (batch_s, HEAD_NUM, group, KEY_DIM)

            out_concat = multi_head_attention_TSP(q, self.k, self.v, group_ninf_mask=group_ninf_mask)
            # shape = (batch_s, group, HEAD_NUM*KEY_DIM)

        # Added layers start
        ###############################################

        residual = out_concat.detach()
        # out_concat = torch.matmul(out_concat.permute(1, 0, 2).unsqueeze(2), self.new)
        # out_concat = out_concat.squeeze(2).permute(1, 0, 2)
        out_concat = F.relu(torch.matmul(out_concat, self.new) + self.new_bias.expand_as(out_concat))
        out_concat = torch.matmul(out_concat, self.new_2) + self.new_bias_2.expand_as(out_concat)
        out_concat += residual

        # Added layers end
        ###############################################

        mh_atten_out = self.multi_head_combine(out_concat)
        # shape = (batch_s, group, EMBEDDING_DIM)

        #  Single-Head Attention, for probability calculation
        #######################################################
        score = torch.matmul(mh_atten_out, self.single_head_key.detach())
        # shape = (batch_s, group, TSP_SIZE)

        score_scaled = score / np.sqrt(EMBEDDING_DIM)
        # shape = (batch_s, group, TSP_SIZE)

        score_clipped = LOGIT_CLIPPING * torch.tanh(score_scaled)

        score_masked = score_clipped + group_ninf_mask.clone()

        probs = F.softmax(score_masked, dim=2)
        # shape = (batch_s, group, TSP_SIZE)

        return probs


def replace_decoder(grouped_actor, batch_s, state, problem):
    """Function to add layers to pretrained model while retaining weights from other layers."""

    # update node_prob_calculator
    if problem == "CVRP":
        raise NotImplementedError("CVRP not implemented")
        grouped_actor.node_prob_calculator = prob_calc_added_layers_CVRP(batch_s)
    elif problem == "TSP":
        grouped_actor.node_prob_calculator = prob_calc_added_layers_TSP(batch_s)
    grouped_actor.node_prob_calculator.load_state_dict(state_dict=state, strict=False)

    return grouped_actor


#######################################################################################################################
# Search process
#######################################################################################################################

def run_eas_lay_decoder(grouped_actor, instance_data, problem_size, config, eval_opts):
    """
    Efficient active search using added layer updates
    """

    dataset_size = len(instance_data[0])

    assert eval_opts.batch_size <= dataset_size

    # Save the original state_dict of the decoder
    original_decoder_state_dict = grouped_actor.node_prob_calculator.state_dict()

    instance_solutions = torch.zeros(dataset_size, problem_size * 2, dtype=torch.int)
    instance_costs = np.zeros((dataset_size))

    if config.problem == "tsp":
        pass
        # from source.tsp.env import GROUP_ENVIRONMENT
    elif config.problem == "cvrp":
        raise NotImplementedError("CVRP not implemented")
        from source.cvrp.env import GROUP_ENVIRONMENT

    for episode in tqdm(range(math.ceil(dataset_size / eval_opts.batch_size))):

        # Load the instances
        ###############################################

        episode_data = get_episode_data(instance_data, episode * eval_opts.batch_size, eval_opts.batch_size, problem_size)
        batch_size = episode_data[0].shape[0]  # Number of instances considered in this iteration

        batch_r = batch_size * p_runs  # Search runs per batch
        batch_s = AUG_S * batch_r  # Model batch size (nb. of instances * the number of augmentations * p_runs)
        group_s = problem_size + 1  # Number of different rollouts per instance (+1 for incumbent solution construction)

        with torch.no_grad():
            aug_data = augment_and_repeat_episode_data(episode_data, problem_size, p_runs, AUG_S)
            env = GROUP_ENVIRONMENT(aug_data, problem_size)

            # Replace the decoder of the loaded model with the modified decoder with added layers
            grouped_actor_modified = replace_decoder(grouped_actor, batch_s, original_decoder_state_dict,
                                                     config.problem).cuda()

            group_state, reward, done = env.reset(group_size=group_s)
            grouped_actor_modified.reset(group_state)  # Generate the embeddings

        # Only update the weights of the added layer during training
        optimizer = optim.Adam(
            [grouped_actor_modified.node_prob_calculator.new, grouped_actor_modified.node_prob_calculator.new_2,
             grouped_actor_modified.node_prob_calculator.new_bias,
             grouped_actor_modified.node_prob_calculator.new_bias_2], lr=param_lr,
            weight_decay=ACTOR_WEIGHT_DECAY)

        incumbent_solutions = torch.zeros(batch_size, problem_size * 2, dtype=torch.int)

        # Start the search
        ###############################################

        t_start = time.time()
        for iter in range(max_iter):
            group_state, reward, done = env.reset(group_size=group_s)

            incumbent_solutions_expanded = incumbent_solutions.repeat(AUG_S, 1).repeat(p_runs, 1)

            # Start generating batch_s * group_s solutions
            ###############################################
            solutions = []

            step = 0
            if config.problem == "cvrp":
                raise NotImplementedError("CVRP not implemented")
                # First Move is given
                first_action = LongTensor(np.zeros((batch_s, group_s)))  # start from node_0-depot
                group_state, reward, done = env.step(first_action)
                solutions.append(first_action.unsqueeze(2))
                step += 1

            # First/Second Move is given
            second_action = LongTensor(np.arange(group_s) % problem_size)[None, :].expand(batch_s, group_s).clone()

            if iter > 0:
                second_action[:, -1] = incumbent_solutions_expanded[:,
                                       step]  # Teacher forcing the imitation learning loss
            group_state, reward, done = env.step(second_action)
            solutions.append(second_action.unsqueeze(2))
            step += 1

            group_prob_list = Tensor(np.zeros((batch_s, group_s, 0)))
            while not done:
                action_probs = grouped_actor_modified.get_action_probabilities(group_state)
                # shape = (batch_s, group_s, problem)
                action = action_probs.reshape(batch_s * group_s, -1).multinomial(1) \
                    .squeeze(dim=1).reshape(batch_s, group_s)
                # shape = (batch_s, group_s)
                if iter > 0:
                    action[:, -1] = incumbent_solutions_expanded[:, step]  # Teacher forcing the imitation learning loss

                if config.problem == "cvrp":
                    action[group_state.finished] = 0  # stay at depot, if you are finished
                group_state, reward, done = env.step(action)
                solutions.append(action.unsqueeze(2))

                batch_idx_mat = torch.arange(int(batch_s))[:, None].expand(batch_s,
                                                                           group_s)
                group_idx_mat = torch.arange(group_s)[None, :].expand(batch_s, group_s)
                chosen_action_prob = action_probs[batch_idx_mat, group_idx_mat, action].reshape(batch_s, group_s)
                # shape = (batch_s, group_s)
                if config.problem == "cvrp":
                    chosen_action_prob[group_state.finished] = 1  # done episode will gain no more probability
                group_prob_list = torch.cat((group_prob_list, chosen_action_prob[:, :, None]), dim=2)
                step += 1

            # Solution generation finished. Update incumbent solutions and best rewards
            ###############################################

            group_reward = reward.reshape(AUG_S, batch_r, group_s)
            solutions = torch.cat(solutions, dim=2)
            if eval_opts.batch_size == 1:
                # Single instance search. Only a single incumbent solution exists that needs to be updated
                max_idx = torch.argmax(reward)
                best_solution_iter = solutions.reshape(-1, solutions.shape[2])
                best_solution_iter = best_solution_iter[max_idx]
                incumbent_solutions[0, :best_solution_iter.shape[0]] = best_solution_iter
                max_reward = reward.max()

            else:
                # Batch search. Update incumbent etc. separately for each instance
                max_reward, _ = group_reward.max(dim=2)
                max_reward, _ = max_reward.max(dim=0)

                reward_g = group_reward.permute(1, 0, 2).reshape(batch_r, -1)
                iter_max_k, iter_best_k = torch.topk(reward_g, k=1, dim=1)
                solutions = solutions.reshape(AUG_S, batch_r, group_s, -1)
                solutions = solutions.permute(1, 0, 2, 3).reshape(batch_r, AUG_S * group_s, -1)
                best_solutions_iter = torch.gather(solutions, 1,
                                                   iter_best_k.unsqueeze(2).expand(-1, -1, solutions.shape[2])).squeeze(
                    1)
                incumbent_solutions[:, :best_solutions_iter.shape[1]] = best_solutions_iter

            # LEARNING - Actor
            # Use the same reinforcement learning method as during the training of the model
            # to update only the weights of the newly added layers
            ###############################################
            group_reward = reward[:, :group_s - 1]
            # shape = (batch_s, group_s - 1)
            group_log_prob = group_prob_list.log().sum(dim=2)
            # shape = (batch_s, group_s)

            group_advantage = group_reward - group_reward.mean(dim=1, keepdim=True)

            group_loss = -group_advantage * group_log_prob[:, :group_s - 1]
            # shape = (batch_s, group_s - 1)
            loss_1 = group_loss.mean()  # Reinforcement learning loss
            loss_2 = -group_log_prob[:, group_s - 1].mean()  # Imitation learning loss
            loss = loss_1 + loss_2 * param_lambda

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(max_reward)

            if time.time() - t_start > max_runtime:
                break

        # Store incumbent solutions and their objective function value
        instance_solutions[episode * eval_opts.batch_size: episode * eval_opts.batch_size + batch_size] = incumbent_solutions
        instance_costs[
        episode * eval_opts.batch_size: episode * eval_opts.batch_size + batch_size] = -max_reward.cpu().numpy()

    return instance_costs, instance_solutions