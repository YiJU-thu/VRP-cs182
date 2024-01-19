import torch
import numpy as np
from torch import nn
from torchvision.ops import MLP
import math

from loguru import logger


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
            encode_edge_matrix=False
    ):
        super(MultiHeadAttention, self).__init__()

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
        V = torch.matmul(hflat, self.W_val).view(shp)

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
    ):
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
            rank_k_approx=0,
            n_edge_encode_layers=0,
            rescale_dist=False,
            node_dim=None,
            normalization='batch',
            feed_forward_hidden=512
    ):
        super(GraphAttentionEncoder, self).__init__()

        # To map input to embedding space
        self.init_embed = nn.Linear(node_dim, embed_dim) if node_dim is not None else None
        
        # change squential model to a list of layers
        # FIXME: an ugly way to avoid problem when loading previous models
        # if n_edge_encode_layers <= 1:
        self.layers = nn.Sequential(*(
            MultiHeadAttentionLayer(n_heads, embed_dim, feed_forward_hidden=feed_forward_hidden, 
                                    normalization=normalization, encode_edge_matrix=(i < n_edge_encode_layers))
            for i in range(n_layers)
        ))
        # else:
        #     self.layers = [
        #         MultiHeadAttentionLayer(n_heads, embed_dim, feed_forward_hidden=feed_forward_hidden, 
        #                                 normalization=normalization, encode_edge_matrix=(i < n_edge_encode_layers))
        #         for i in range(n_layers)
        #     ]

        self.rescale_dist = rescale_dist
        self.rank_k_approx = rank_k_approx
        self.non_Euc = non_Euc
        self.n_edge_encode_layers = n_edge_encode_layers

        if rescale_dist:
            scale_factors_dim = 3 if non_Euc else 1
        else:
            scale_factors_dim = 0        
        graph_embed_layers = 3  # TODO: make this a parameter
        add_graph_dim = rank_k_approx + scale_factors_dim
        self.graph_embed = MLP(embed_dim+add_graph_dim, [embed_dim for _ in range(graph_embed_layers)])

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

        # the embedding of the graph is the concatenation of the average of h, the first k singular values
        # (and scale_dist if rescale_dist is True) going through a linear layer
        # (batch_size, embed_dim)
        if self.rescale_dist:
            assert scale_factors is not None, "rescale_dist=True but scale_dist is None"
            graph_init_embed = torch.cat([h.mean(dim=1), S, scale_factors], dim=1)
        else:
            graph_init_embed = torch.cat([h.mean(dim=1), S], dim=1)
        graph_embedding = self.graph_embed(graph_init_embed)
            

        return (
            h,  # (batch_size, graph_size, embed_dim)
            graph_embedding # (batch_size, embed_dim)
        )
