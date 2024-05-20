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

use_cuda = torch.cuda.is_available()

FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
BoolTensor = torch.cuda.BoolTensor if use_cuda else torch.BoolTensor
Tensor = FloatTensor

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

    def __init__(self, data, problem_size):
        # seq.shape = (batch, TSP_SIZE, 2)

        self.data = data[0]
        self.distance = data[1]
        self.batch_s = self.data.size(0)
        self.group_s = None
        self.group_state = None
        self.problem_size = problem_size

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

        print('111111',self.distance.size() == group_travel_distances.size())
        return self.distance