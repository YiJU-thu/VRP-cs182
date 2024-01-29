import torch
import torch.nn.functional as F
import torch.nn as nn

from models.gcn_layers import ResidualGatedGCNLayer, MLP
from gcn_utils.model_utils import *

import os, sys
curr_dir = os.path.dirname(os.path.realpath(__file__))
root_dir = os.path.join(curr_dir, '../../../')
if root_dir not in sys.path:
    sys.path.append(root_dir)
am_dir = os.path.join(root_dir, 'AM2019')
if am_dir not in sys.path:
    sys.path.append(am_dir)
from AM2019.nets.attention_model import AttentionModel
from AM2019.problems import TSP
from utils_project.utils_vrp import normalize_graph, recover_graph


class AttModel(nn.Module):

    def __init__(self, config, dtypeFloat, dtypeLong):
        super(AttModel, self).__init__()
        self.dtypeFloat = dtypeFloat
        self.dtypeLong = dtypeLong
        # Define net parameters
        
        self.encoder = AttentionModel(
            config.embedding_dim,
            config.hidden_dim,
            problem=TSP,    # FIXME
            non_Euc=True,
            rank_k_approx=config.rank_k_approx,
            rescale_dist=config.rescale_dist,
            svd_original_edge=config.svd_original_edge,
            mul_sigma_uv=config.mul_sigma_uv,
            full_svd=config.full_svd,
            only_distance=config.only_distance,
            n_edge_encode_layers=config.n_edge_encode_layers,
            encode_original_edge=config.encode_original_edge,
            # update_context_node=config.update_context_node,
            # aug_graph_embed=config.aug_graph_embed,
            n_encode_layers=config.n_encode_layers,
            # mask_inner=True,
            # mask_logits=True,
            normalization=config.normalization,
            tanh_clipping=config.tanh_clipping,
            # checkpoint_encoder=config.checkpoint_encoder,
            # shrink_size=config.shrink_size,
            encoder_only=True
        )
    
    def forward(self, x_edges, x_edges_values, x_rel_edges_values, x_scale_factors, x_nodes, x_nodes_coord, y_edges,
                edge_cw, num_neg = 4, loss_type = "CE", gamma = 1):
        """
        Args:
            x_edges: Input edge adjacency matrix (batch_size, num_nodes, num_nodes)
            x_edges_values: Input edge distance matrix (batch_size, num_nodes, num_nodes)
            x_nodes: Input nodes (batch_size, num_nodes)
            x_nodes_coord: Input node coordinates (batch_size, num_nodes, node_dim)
            y_edges: Targets for edges (batch_size, num_nodes, num_nodes)
            edge_cw: Class weights for edges loss
            # y_nodes: Targets for nodes (batch_size, num_nodes, num_nodes)
            # node_cw: Class weights for nodes loss

        Returns:
            y_pred_edges: Predictions for edges (batch_size, num_nodes, num_nodes)
            # y_pred_nodes: Predictions for nodes (batch_size, num_nodes)
            loss: Value of loss function
        """
        
        input = {
            "coords": x_nodes_coord,
            "distance": x_edges_values,
            "rel_distance": x_rel_edges_values, 
            "scale_factors": x_scale_factors, 
        }
        
        # Node and edge embedding
        u_mat = self.encoder(input)
        
        # MLP classifiers
        y_pred_edges = u_mat  # B x V x V x voc_edges_out
        
        # Compute loss
        edge_cw = torch.Tensor(edge_cw).type(self.dtypeFloat)  # Convert to tensors
        loss = loss_edges(y_pred_edges, y_edges, edge_cw,
                          loss_type = loss_type, gamma = gamma)
        
        return y_pred_edges, loss