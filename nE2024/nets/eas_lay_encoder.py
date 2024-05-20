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

# 