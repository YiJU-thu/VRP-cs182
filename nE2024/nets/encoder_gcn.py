import torch
import numpy as np
from torch import nn
from torch.nn import DataParallel
from torch.utils.checkpoint import checkpoint
import torch.nn.functional as F
from torchvision.ops import MLP
import math

from nets.init_embed import InitEncoder

from loguru import logger

# NOTE: a lot are shared with gat encoder, consider refactoring later

class GCNEncoder(nn.Module):

    def __init__(self,
                 embedding_dim,
                 hidden_dim,
                 problem,
                 non_Euc=False,
                 rank_k_approx=0,
                 svd_original_edge=False,
                 mul_sigma_uv=False,
                 full_svd=False,
                 only_distance=False,
                 encode_original_edge=False,  # unused!
                 rescale_dist=False,
                 n_encode_layers=2,
                 normalization='batch', # unused!
                 checkpoint_encoder=False,  # unused!
                 return_heatmap=False,
                 umat_embed_layers=3,
                 aug_graph_embed_layers=3,
                 gcn_aggregation="sum",
                 edge_embedding_dim=None,
                 adj_mat_embedding_dim=None,
                 kNN=20,
                 ):
        super(GCNEncoder, self).__init__()

        self.return_heatmap = return_heatmap

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_encode_layers = n_encode_layers
        self.decode_type = None
        self.temp = 1.0
        self.allow_partial = problem.NAME == 'sdvrp'
        self.is_vrp = problem.NAME == 'cvrp' or problem.NAME == 'sdvrp'
        self.is_orienteering = problem.NAME == 'op'
        self.is_pctsp = problem.NAME == 'pctsp'

        self.problem = problem
        self.checkpoint_encoder = checkpoint_encoder

        self.non_Euc = non_Euc
        # idea 1: node feature augmentation
        self.rank_k_approx = rank_k_approx
        self.svd_original_edge = svd_original_edge
        self.mul_sigma_uv = mul_sigma_uv
        self.full_svd = full_svd
        self.only_distance = only_distance
        # idea 2: edge feature encoding
        self.encode_original_edge = encode_original_edge
        self.rescale_dist = rescale_dist
        self.aug_graph_embed = True

        if encode_original_edge:
            assert non_Euc == True, "edge encoding is only supported for non-Euclidean input"

        self.init_embed = InitEncoder(
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            problem=problem,
            non_Euc=non_Euc,
            rank_k_approx=rank_k_approx,
            svd_original_edge=svd_original_edge,
            mul_sigma_uv=mul_sigma_uv,
            full_svd=full_svd,
            only_distance=only_distance,
            edge_embedding_dim=embedding_dim//2,
            adj_mat_embedding_dim=embedding_dim//2,
            kNN=kNN,
        )

        self.embedder = ResidualGatedGCNModel(
            embed_dim=embedding_dim,
            n_layers=self.n_encode_layers,
            non_Euc=non_Euc,
            dim_Sk = rank_k_approx * (not mul_sigma_uv),    # if mul_sigma_uv, no longer consider Sk at graph feature level
            # normalization=normalization,
            rescale_dist=rescale_dist,
            return_u_mat=self.return_heatmap, # if True, return a compatibility matrix; else, return node embeddings & graph embedding
            umat_embed_layers=umat_embed_layers,
            aug_graph_embed_layers=aug_graph_embed_layers,
            gcn_aggregation=gcn_aggregation
        )


    def forward(self, input):
        """
        :param input: (batch_size, graph_size, node_dim) input node features or dictionary with multiple tensors
        :return:
        """
        I, N, _ = input['coords'].shape
        scale_factors_dim = 2 * self.non_Euc + 1
        if self.rescale_dist:
            assert input['scale_factors'].shape == (I, scale_factors_dim), "scale_dist must be of shape (I, scale_factors_dim)"
        else:
            assert input.get('scale_factors') is None, "scale_dist must be None if rescale_dist is False, but get {}".format(input.get('scale_factors'))

        # if return_heatmap, return a compatibility matrix
        # else, return node embeddings & graph embedding

        assert self.checkpoint_encoder == False, "hasn't implemented checkpoint :("
        if self.checkpoint_encoder and self.training:  # Only checkpoint if we need gradients
            return checkpoint(self.embedder, self._init_embed(input))
        else:
            # init_embed: (batch_size, graph_size, embedding_dim) - h^0_i's
            # S: (batch_size, graph_size, rank_k_approx) - first k singular values of the (rel) distance matrix
            init_embed, edge_embed, S = self.init_embed(input)
            
            scale_factors = None if not self.rescale_dist else input['scale_factors']
            
            # if self.encoder_only: return u_mat
            # else: return (embeddings, graph_embed)
            return self.embedder(init_embed, edge_embed, S, scale_factors=scale_factors)


class ResidualGatedGCNModel(nn.Module):
    """Residual Gated GCN Model for outputting predictions as edge adjacency matrices.

    References:
        Paper: https://arxiv.org/pdf/1711.07553v2.pdf
        Code: https://github.com/xbresson/spatial_graph_convnets
    """

    def __init__(
            self, 
            embed_dim,
            n_layers,
            non_Euc=False,
            dim_Sk=0,
            rescale_dist=False,
            aug_graph_embed_layers=3,
            return_u_mat=False,
            umat_embed_layers=3,
            gcn_aggregation="sum"):
        super(ResidualGatedGCNModel, self).__init__()
        
        
        self.return_u_mat = return_u_mat
        # if True, return the last compatability matrix
        # if False, return (node embeddings, graph embedding)
        
        gcn_layers = []
        for layer in range(n_layers):
            gcn_layers.append(ResidualGatedGCNLayer(embed_dim, gcn_aggregation))
        self.layers = nn.ModuleList(gcn_layers)
        
        self.rescale_dist = rescale_dist
        self.non_Euc = non_Euc

        scale_factors_dim = (2 * non_Euc + 1) * rescale_dist 
        add_graph_dim = dim_Sk + scale_factors_dim
        self.add_graph_dim = add_graph_dim
        
        if not return_u_mat:
            if add_graph_dim > 0:
                self.graph_embed = MLP(embed_dim+add_graph_dim, [embed_dim for _ in range(aug_graph_embed_layers)])
        else:
            self.u_mat_embed = MLP(embed_dim+add_graph_dim, [embed_dim for _ in range(umat_embed_layers)]+[1])
        
        
        # self.dtypeFloat = dtypeFloat
        # self.dtypeLong = dtypeLong
        # # Define net parameters
        # self.num_nodes = config.num_nodes
        # self.node_dim = config.node_dim
        # self.voc_nodes_in = config.voc_nodes_in
        # self.voc_nodes_out = config.num_nodes  # config['voc_nodes_out']
        # self.voc_edges_in = config.voc_edges_in
        # self.voc_edges_out = config.voc_edges_out
        # self.hidden_dim = config.hidden_dim
        # self.num_layers = config.num_layers
        # self.mlp_layers = config.mlp_layers
        # self.aggregation = config.aggregation
        # # Node and edge embedding layers/lookups
        # self.nodes_coord_embedding = nn.Linear(self.node_dim, self.hidden_dim, bias=False)
        # self.edges_values_embedding = nn.Linear(1, self.hidden_dim//2, bias=False)
        # self.edges_embedding = nn.Embedding(self.voc_edges_in, self.hidden_dim//2)
        # # Define GCN Layers
        # gcn_layers = []
        # for layer in range(self.num_layers):
        #     gcn_layers.append(ResidualGatedGCNLayer(self.hidden_dim, self.aggregation))
        # self.gcn_layers = nn.ModuleList(gcn_layers)
        # # Define MLP classifiers
        # self.mlp_edges = MLP(self.hidden_dim, self.voc_edges_out, self.mlp_layers)
        # # self.mlp_nodes = MLP(self.hidden_dim, self.voc_nodes_out, self.mlp_layers)

    def forward(self, x, e, S, scale_factors=None, mask=None):
        """
        Args:
            x: initial node embeddings (batch_size, num_nodes, embed_dim)
            e: initial edge embeddings (batch_size, num_nodes, num_nodes, embed_dim)
            S: singular values of the distance matrix (batch_size, num_nodes, rank_k_approx), None if ...
            scale_factors: scale factors for the distance matrix (batch_size, num_nodes, scale_factors_dim), None if ...
            mask: FIXME: no used. may be deleted

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
        # # Node and edge embedding
        # x = self.nodes_coord_embedding(x_nodes_coord)  # B x V x H
        # e_vals = self.edges_values_embedding(x_edges_values.unsqueeze(3))  # B x V x V x H
        # e_tags = self.edges_embedding(x_edges)  # B x V x V x H
        # e = torch.cat((e_vals, e_tags), dim=3)
        
        # # permute kaibin QIU
        # # FIXME: do we need to permute?
        x = x.permute(0, 2, 1) # B x H x V
        e = e.permute(0, 3, 1, 2) # B x H x V x V
        
        # GCN layers
        for l in range(len(self.layers)):
            x, e = self.layers[l](x, e)  # B x V x H, B x V x V x H
        
        h = x.permute(0,2,1) if not self.return_u_mat else e.permute(0,2,3,1)
        
        if self.add_graph_dim > 0:
            S = S if S is not None else torch.zeros(h.size(0), 0, device=h.device)
            scale_factors = scale_factors if self.rescale_dist else torch.zeros(h.size(0), 0, device=h.device)

        
        if not self.return_u_mat:
            # the embedding of the graph is the concatenation of the average of h, the first k singular values
            # (and scale_dist if rescale_dist is True) going through a linear layer
            # (batch_size, embed_dim)
            
            if self.add_graph_dim == 0:
                graph_embedding = h.mean(dim=1)
            else:
                # NOTE: we rarely have separate graph features, so we rarely use this branch
                graph_init_embed = torch.cat([h.mean(dim=1), S, scale_factors], dim=1)
                graph_embedding = self.graph_embed(graph_init_embed)
                
            return {
                "embeddings": h,  # (batch_size, graph_size, embed_dim)
                "graph_embed": graph_embedding # (batch_size, embed_dim)
            }

        else:
            # Here [h] is the last compatibility matrix (attn)
            # (n_heads, I, N, N) -> (I, N, N, n_heads)
            # u_mat = h.permute(1, 2, 3, 0)
            u_mat = h
            I, N, _, embed_dim = u_mat.size()
            # (I, N, N, n_heads) -> (I, N, N, n_heads+add_graph_dim)
            if self.add_graph_dim > 0:
                graph_features = torch.cat([S, scale_factors], dim=1)
                # broadcast graph features to the same shape as u_mat
                graph_features = graph_features.unsqueeze(1).unsqueeze(1).expand(I, N, N, self.add_graph_dim)
                
                u_mat = torch.cat([u_mat, graph_features], dim=3)

            u_mat = self.u_mat_embed(u_mat.view(-1, u_mat.size(-1))).view(*u_mat.size()[:3], -1).squeeze(-1)
            assert u_mat.size() == (I,N,N), f"{u_mat.size()} != [{I}, {N}, {N}]"
            return {"heatmap": u_mat}
        
        
        
        
        
        
        
        # MLP classifier
        y_pred_edges = self.mlp_edges(e)  # B x V x V x voc_edges_out
        # y_pred_nodes = self.mlp_nodes(x)  # B x V x voc_nodes_out
        
        # Compute loss
        edge_cw = torch.Tensor(edge_cw).type(self.dtypeFloat)  # Convert to tensors
        loss = loss_edges(y_pred_edges, y_edges, edge_cw,
                          loss_type = loss_type, gamma = gamma)
        
        return y_pred_edges, loss


class BatchNormNode(nn.Module):
    """Batch normalization for node features.
    """

    def __init__(self, hidden_dim):
        super(BatchNormNode, self).__init__()
        self.batch_norm = nn.BatchNorm1d(hidden_dim, track_running_stats=False)

    def forward(self, x):
        """
        Args:
            x: Node features (batch_size, hidden_dim, num_nodes)

        Returns:
            x_bn: Node features after batch normalization (batch_size, hidden_dim, num_nodes)
        """
        x_bn = self.batch_norm(x) # B x H x N
        return x_bn


class BatchNormEdge(nn.Module):
    """Batch normalization for edge features.
    """

    def __init__(self, hidden_dim):
        super(BatchNormEdge, self).__init__()
        self.batch_norm = nn.BatchNorm2d(hidden_dim, track_running_stats=False)

    def forward(self, e):
        """
        Args:
            e: Edge features (batch_size, hidden_dim, num_nodes, num_nodes)

        Returns:
            e_bn: Edge features after batch normalization (batch_size, hidden_dim, num_nodes, num_nodes)
        """
        e_bn = self.batch_norm(e) # B x H x N x N
        return e_bn


class NodeFeatures(nn.Module):
    """Convnet features for nodes.
    
    Using `sum` aggregation:
        x_i = U*x_i +  sum_j [ gate_ij * (V*x_j) ]
    
    Using `mean` aggregation:
        x_i = U*x_i + ( sum_j [ gate_ij * (V*x_j) ] / sum_j [ gate_ij] )
    """
    
    def __init__(self, hidden_dim, aggregation="mean"):
        super(NodeFeatures, self).__init__()
        self.aggregation = aggregation
        self.U = nn.Conv1d(hidden_dim, hidden_dim, 1)
        self.V = nn.Conv1d(hidden_dim, hidden_dim, 1)

    def forward(self, x, edge_gate):
        """
        Args:
            x: Node features (batch_size, num_nodes, hidden_dim)
            edge_gate: Edge gate values (batch_size, num_nodes, num_nodes, hidden_dim)

        Returns:
            x_new: Convolved node features (batch_size, num_nodes, hidden_dim)
        """
        Ux = self.U(x)  # B x H x V
        Vx = self.V(x)  # B x H x V
        Vx = Vx.unsqueeze(2)  # extend Vx from "B x H x V" to "B x H x 1 x V"
        gateVx = edge_gate * Vx  # B x H x V x V
        if self.aggregation=="mean":
            x_new = Ux + torch.sum(gateVx, dim=3) / (1e-20 + torch.sum(edge_gate, dim=3))  # B x H x V
        elif self.aggregation=="sum":
            x_new = Ux + torch.sum(gateVx, dim=3)  # B x H x V
        return x_new


class EdgeFeatures(nn.Module):
    """Convnet features for edges.

    e_ij = U*e_ij + V*(x_i + x_j)
    """

    def __init__(self, hidden_dim):
        super(EdgeFeatures, self).__init__()
        self.U = nn.Conv2d(hidden_dim, hidden_dim, (1,1))
        self.V = nn.Conv1d(hidden_dim, hidden_dim, 1)
        
    def forward(self, x, e):
        """
        Args:
            x: Node features (batch_size, num_nodes, hidden_dim)
            e: Edge features (batch_size, num_nodes, num_nodes, hidden_dim)

        Returns:
            e_new: Convolved edge features (batch_size, num_nodes, num_nodes, hidden_dim)
        """
        Ue = self.U(e) # B x H x V x V
        Vx = self.V(x) # B x H x V
        Wx = Vx.unsqueeze(2)  # Extend Vx from "B x H x V" to "B x H x 1 x V"
        Vx = Vx.unsqueeze(3)  # extend Vx from "B x H x V" to "B x H x V x 1"
        e_new = Ue + Vx + Wx
        return e_new


class ResidualGatedGCNLayer(nn.Module):
    """Convnet layer with gating and residual connection.
    """

    def __init__(self, hidden_dim, aggregation="sum"):
        super(ResidualGatedGCNLayer, self).__init__()
        self.node_feat = NodeFeatures(hidden_dim, aggregation)
        self.edge_feat = EdgeFeatures(hidden_dim)
        self.bn_node = BatchNormNode(hidden_dim)
        self.bn_edge = BatchNormEdge(hidden_dim)

    def forward(self, x, e):
        """
        Args:
            x: Node features (batch_size, hidden_dim, num_nodes)
            e: Edge features (batch_size, hidden_dim, num_nodes, num_nodes)

        Returns:
            x_new: Convolved node features (batch_size, hidden_dim, num_nodes)
            e_new: Convolved edge features (batch_size, hidden_dim, num_nodes, num_nodes)
        """
        e_in = e # B x H x V x V
        x_in = x # B x H x V
        # Edge convolution
        e_tmp = self.edge_feat(x_in, e_in)  # B x H x V x V
        # Compute edge gates
        edge_gate = F.sigmoid(e_tmp)
        # Node convolution
        x_tmp = self.node_feat(x_in, edge_gate) # B x H x V
        # Batch normalization
        e_tmp = self.bn_edge(e_tmp)
        x_tmp = self.bn_node(x_tmp)
        # ReLU Activation
        e = F.relu(e_tmp)
        x = F.relu(x_tmp)
        # Residual connection
        x_new = x_in + x
        e_new = e_in + e
        return x_new, e_new


# class MLP(nn.Module):
#     """Multi-layer Perceptron for output prediction.
#     """

#     def __init__(self, hidden_dim, output_dim, L=2):
#         super(MLP, self).__init__()
#         self.L = L
#         U = []
#         for layer in range(self.L - 1):
#             U.append(nn.Conv2d(hidden_dim, hidden_dim, (1, 1))) # B x H x V x V
#         self.U = nn.ModuleList(U)
#         self.V = nn.Conv2d(hidden_dim, output_dim, (1, 1)) # B x O x V x V

#     def forward(self, x):
#         """
#         Args:
#             x: Input features (batch_size, hidden_dim, num_nodes, num_nodes)

#         Returns:
#             y: Output predictions (batch_size, output_dim, num_nodes, num_nodes)
#         """
#         Ux = x
#         for U_i in self.U:
#             Ux = U_i(Ux)  # B x H x V x V
#             Ux = F.relu(Ux)  # B x H x V x V
#         y = self.V(Ux)  # B x O x V x V
#         y = y.permute(0, 2, 3, 1)
#         return y