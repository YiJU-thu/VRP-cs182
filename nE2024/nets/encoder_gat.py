import torch
import numpy as np
from torch import nn
from torch.nn import DataParallel
from torch.utils.checkpoint import checkpoint
from torchvision.ops import MLP
import math

from nets.init_embed import InitEncoder

from loguru import logger


class AttentionEncoder(nn.Module):

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
                 n_edge_encode_layers=0,
                 encode_original_edge=False,
                 rescale_dist=False,
                 n_encode_layers=2,
                 normalization='batch',
                 n_heads=8,
                 checkpoint_encoder=False,
                 return_heatmap=False,
                 umat_embed_layers=3,
                 aug_graph_embed_layers=3,
                 no_coords=False,
                 random_node_dim=0
                 ):
        super(AttentionEncoder, self).__init__()

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
        self.n_heads = n_heads
        self.checkpoint_encoder = checkpoint_encoder

        self.non_Euc = non_Euc
        # idea 1: node feature augmentation
        self.rank_k_approx = rank_k_approx
        self.svd_original_edge = svd_original_edge
        self.mul_sigma_uv = mul_sigma_uv
        self.full_svd = full_svd
        self.only_distance = only_distance
        # idea 2: edge feature encoding
        self.n_edge_encode_layers = n_edge_encode_layers
        self.encode_original_edge = encode_original_edge
        self.rescale_dist = rescale_dist
        self.aug_graph_embed = True

        assert n_edge_encode_layers <= n_encode_layers, "n_edge_encode_layer must be <= n_encode_layers"
        if return_heatmap:
            assert n_edge_encode_layers <= n_encode_layers - 1, "no mix scores at the last layer" # FIXME
        if n_edge_encode_layers > 0:
            assert non_Euc == True, "edge encoding is only supported for non-Euclidean input"
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
            no_coords=no_coords,
            random_node_dim=random_node_dim
        )

        self.embedder = GraphAttentionEncoder(
            n_heads=n_heads,
            embed_dim=embedding_dim,
            n_layers=self.n_encode_layers,
            non_Euc=non_Euc,
            dim_Sk = rank_k_approx * (not mul_sigma_uv),    # if mul_sigma_uv, no longer consider Sk at graph feature level
            n_edge_encode_layers=n_edge_encode_layers, 
            normalization=normalization,
            rescale_dist=rescale_dist,
            return_u_mat=self.return_heatmap, # if True, return a compatibility matrix; else, return node embeddings & graph embedding
            umat_embed_layers=umat_embed_layers,
            aug_graph_embed_layers=aug_graph_embed_layers
        )


    def forward(self, input):
        """
        :param input: (batch_size, graph_size, node_dim) input node features or dictionary with multiple tensors
        :return:
        """
        I, N = self.init_embed._get_input_shape(input)
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
            init_embed, S = self.init_embed(input)
            
            if self.n_edge_encode_layers > 0:
                edge_matrix = input['distance'] if self.encode_original_edge else input['rel_distance']
            else:
                edge_matrix = None
            
            scale_factors = None if not self.rescale_dist else input['scale_factors']
            
            # if self.encoder_only: return u_mat
            # else: return (embeddings, graph_embed)
            return self.embedder(init_embed, S, scale_factors=scale_factors, edge_matrix=edge_matrix)
  

class SkipConnection(nn.Module):

    def __init__(self, module):
        super(SkipConnection, self).__init__()
        self.module = module

    def forward(self, input):
        if isinstance(input, tuple):
            input, kwargs = input
        else:
            kwargs = {}
        return input + self.module(input, **kwargs)


class MultiHeadAttention(nn.Module):
    def __init__(
            self,
            n_heads,
            input_dim,
            embed_dim,
            val_dim=None,
            key_dim=None,
            encode_edge_matrix=False,
            return_u_mat=False
    ):
        super(MultiHeadAttention, self).__init__()

        self.return_u_mat = return_u_mat
        
        if val_dim is None:
            val_dim = embed_dim // n_heads
        if key_dim is None:
            key_dim = val_dim

        self.n_heads = n_heads
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.val_dim = val_dim
        self.key_dim = key_dim

        self.norm_factor = 1 / math.sqrt(key_dim)  # See Attention is all you need

        self.W_query = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        self.W_key = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        
        if not return_u_mat:
            self.W_val = nn.Parameter(torch.Tensor(n_heads, input_dim, val_dim))
            self.W_out = nn.Parameter(torch.Tensor(n_heads, val_dim, embed_dim))

        self.encode_edge_matrix = encode_edge_matrix
        if encode_edge_matrix:
            # FIXME: this MLP structure is arbitrarily chosen
            self.edge_mlp = MLP(1, [16, 16, n_heads])
        else:
            self.edge_mlp = None

        self.init_parameters()

    def init_parameters(self):

        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, q, h=None, mask=None, edge_matrix=None):
        """

        :param q: queries (batch_size, n_query, input_dim)
        :param h: data (batch_size, graph_size, input_dim)
        :param mask: mask (batch_size, n_query, graph_size) or viewable as that (i.e. can be 2 dim if n_query == 1)
        Mask should contain 1 if attention is not possible (i.e. mask is negative adjacency)
        :return:
        """
        if h is None:
            h = q  # compute self-attention

        # h should be (batch_size, graph_size, input_dim)
        batch_size, graph_size, input_dim = h.size()
        n_query = q.size(1)
        assert q.size(0) == batch_size
        assert q.size(2) == input_dim
        assert input_dim == self.input_dim, "Wrong embedding dimension of input"

        hflat = h.contiguous().view(-1, input_dim)
        qflat = q.contiguous().view(-1, input_dim)

        # last dimension can be different for keys and values
        shp = (self.n_heads, batch_size, graph_size, -1)
        shp_q = (self.n_heads, batch_size, n_query, -1)

        # Calculate queries, (n_heads, n_query, graph_size, key/val_size)
        Q = torch.matmul(qflat, self.W_query).view(shp_q)
        # Calculate keys and values (n_heads, batch_size, graph_size, key/val_size)
        K = torch.matmul(hflat, self.W_key).view(shp)

        # Calculate compatibility (n_heads, batch_size, n_query, graph_size)
        compatibility = self.norm_factor * torch.matmul(Q, K.transpose(2, 3))

        # TODO: add edge matrix encodings here
        if self.encode_edge_matrix:
            I, N, _ = edge_matrix.size()
            edge_matrix_flat = edge_matrix.view(-1, 1)  
            edge_matrix_processed = self.edge_mlp(edge_matrix_flat) 
            edge_matrix_processed = edge_matrix_processed.view(I, N, N, self.n_heads).permute(3, 0, 1, 2)
            assert edge_matrix_processed.size() == compatibility.size(), f"{edge_matrix_processed.size()} != {compatibility.size()}"
            compatibility = compatibility + edge_matrix_processed


        # Optionally apply mask to prevent attention
        if mask is not None:
            mask = mask.view(1, batch_size, n_query, graph_size).expand_as(compatibility)
            compatibility[mask] = -np.inf

        attn = torch.softmax(compatibility, dim=-1)

        # If there are nodes with no neighbours then softmax returns nan so we fix them to 0
        if mask is not None:
            attnc = attn.clone()
            attnc[mask] = 0
            attn = attnc
        
        if self.return_u_mat:
            return attn

        V = torch.matmul(hflat, self.W_val).view(shp)

        heads = torch.matmul(attn, V)

        out = torch.mm(
            heads.permute(1, 2, 0, 3).contiguous().view(-1, self.n_heads * self.val_dim),
            self.W_out.view(-1, self.embed_dim)
        ).view(batch_size, n_query, self.embed_dim)

        # Alternative:
        # headst = heads.transpose(0, 1)  # swap the dimensions for batch and heads to align it for the matmul
        # # proj_h = torch.einsum('bhni,hij->bhnj', headst, self.W_out)
        # projected_heads = torch.matmul(headst, self.W_out)
        # out = torch.sum(projected_heads, dim=1)  # sum across heads

        # Or:
        # out = torch.einsum('hbni,hij->bnj', heads, self.W_out)

        return out


class Normalization(nn.Module):

    def __init__(self, embed_dim, normalization='batch'):
        super(Normalization, self).__init__()

        normalizer_class = {
            'batch': nn.BatchNorm1d,
            'instance': nn.InstanceNorm1d
        }.get(normalization, None)

        self.normalizer = normalizer_class(embed_dim, affine=True)

        # Normalization by default initializes affine parameters with bias 0 and weight unif(0,1) which is too large!
        # self.init_parameters()

    def init_parameters(self):

        for name, param in self.named_parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, input):

        if isinstance(self.normalizer, nn.BatchNorm1d):
            return self.normalizer(input.view(-1, input.size(-1))).view(*input.size())
        elif isinstance(self.normalizer, nn.InstanceNorm1d):
            return self.normalizer(input.permute(0, 2, 1)).permute(0, 2, 1)
        else:
            assert self.normalizer is None, "Unknown normalizer type"
            return input


class MultiHeadAttentionLayer(nn.Sequential):

    def __init__(
            self,
            n_heads,
            embed_dim,
            encode_edge_matrix=False,
            feed_forward_hidden=512,
            normalization='batch',
            return_u_mat=False
    ):
        if return_u_mat:    # return the compatibility matrix
            super(MultiHeadAttentionLayer, self).__init__(
                MultiHeadAttention(
                    n_heads,
                    input_dim=embed_dim,
                    embed_dim=embed_dim,
                    encode_edge_matrix=encode_edge_matrix,
                    return_u_mat=True
                )
            )
            return

        super(MultiHeadAttentionLayer, self).__init__(
            SkipConnection(
                MultiHeadAttention(
                    n_heads,
                    input_dim=embed_dim,
                    embed_dim=embed_dim,
                    encode_edge_matrix=encode_edge_matrix
                )
            ),
            Normalization(embed_dim, normalization),
            SkipConnection(
                nn.Sequential(
                    nn.Linear(embed_dim, feed_forward_hidden),
                    nn.ReLU(),
                    nn.Linear(feed_forward_hidden, embed_dim)
                ) if feed_forward_hidden > 0 else nn.Linear(embed_dim, embed_dim)
            ),
            Normalization(embed_dim, normalization)
        )
    



class GraphAttentionEncoder(nn.Module):
    def __init__(
            self,
            n_heads,
            embed_dim,
            n_layers,
            non_Euc=False,
            dim_Sk=0,
            n_edge_encode_layers=0,
            rescale_dist=False,
            aug_graph_embed_layers=3,
            node_dim=None,
            normalization='batch',
            feed_forward_hidden=512,
            return_u_mat=False,
            umat_embed_layers=3
    ):
        super(GraphAttentionEncoder, self).__init__()

        self.return_u_mat = return_u_mat
        # if True, return the last compatability matrix
        # if False, return (node embeddings, graph embedding)

        # To map input to embedding space
        self.init_embed = nn.Linear(node_dim, embed_dim) if node_dim is not None else None
        
        # change squential model to a list of layers
        # FIXME: an ugly way to avoid problem when loading previous models
        # if n_edge_encode_layers <= 1:
        self.layers = nn.Sequential(*(
            MultiHeadAttentionLayer(n_heads, embed_dim, feed_forward_hidden=feed_forward_hidden, 
                                    normalization=normalization, encode_edge_matrix=(i < n_edge_encode_layers),
                                    return_u_mat=((i == n_layers-1) and return_u_mat))
            for i in range(n_layers)
        ))
        
        self.rescale_dist = rescale_dist
        self.non_Euc = non_Euc
        self.n_edge_encode_layers = n_edge_encode_layers


        scale_factors_dim = (2 * non_Euc + 1) * rescale_dist 
        add_graph_dim = dim_Sk + scale_factors_dim
        self.add_graph_dim = add_graph_dim
        
        if not return_u_mat:
            if add_graph_dim > 0:
                dim = embed_dim+add_graph_dim
                self.graph_embed = MLP(dim, [2*dim for _ in range(aug_graph_embed_layers)])
        else:
            dim = n_heads+add_graph_dim
            self.u_mat_embed = MLP(dim, [2*dim for _ in range(umat_embed_layers)]+[1])
            # self.u_mat_embed = MLP(n_heads+add_graph_dim, [embed_dim for _ in range(umat_embed_layers)]+[1])  # this will increase huge memory usage!

    def forward(self, x, S, scale_factors=None, mask=None, edge_matrix=None):

        assert mask is None, "TODO mask not yet supported!"

        # Yifan TODO: rewrite this part to add singular values of the distance matrix to the graph embedding
        
        # Batch multiply to get initial embeddings of nodes
        h = self.init_embed(x.view(-1, x.size(-1))).view(*x.size()[:2], -1) if self.init_embed is not None else x

        # FIXME: it's wired to pass this in order to avoid a Sequential forward error
        # Now allows pass edge_matrix to multiple encoding layers
        # if isinstance(self.layers, list):
        for l in range(len(self.layers)):
            if l < self.n_edge_encode_layers:
                h = self.layers[l]((h, {"edge_matrix":edge_matrix}))
            else:
                h = self.layers[l](h)
        # else:   # previously trained models are saved as sequential
        #     assert isinstance(self.layers, nn.Sequential)
        #     h = self.layers[0]((h, {"edge_matrix":edge_matrix}))

        
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
            u_mat = h.permute(1, 2, 3, 0)
            I, N, _, n_heads = u_mat.size()
            # (I, N, N, n_heads) -> (I, N, N, n_heads+add_graph_dim)
            if self.add_graph_dim > 0:
                graph_features = torch.cat([S, scale_factors], dim=1)
                # broadcast graph features to the same shape as u_mat
                graph_features = graph_features.unsqueeze(1).unsqueeze(1).expand(I, N, N, self.add_graph_dim)
                
                u_mat = torch.cat([u_mat, graph_features], dim=3)

            u_mat = self.u_mat_embed(u_mat.view(-1, u_mat.size(-1))).view(*u_mat.size()[:3], -1).squeeze(-1)
            assert u_mat.size() == (I,N,N), f"{u_mat.size()} != [{I}, {N}, {N}]"
            return {"heatmap": u_mat}
