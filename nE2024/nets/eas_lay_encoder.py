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

from nets.encoder_gat import GraphAttentionEncoder
from nets.encoder_gcn import ResidualGatedGCNModel
from utils.tensor_functions import recover_full_edge_mat



use_cuda = torch.cuda.is_available()

def get_encoder_parameters_gat(encoder):
    # Extract parameters necessary for initializing GraphAttentionEncoder
    embedder_params = {
        'n_heads': encoder.n_heads,
        'embed_dim': encoder.embedding_dim,
        'n_layers': encoder.n_encode_layers,
        'non_Euc': encoder.non_Euc,
        'dim_Sk': encoder.rank_k_approx * (not encoder.mul_sigma_uv),
        'n_edge_encode_layers': encoder.n_edge_encode_layers,
        'rescale_dist': encoder.rescale_dist,
        'normalization': 'batch', ##### This is hard coded for now, should be changed from the encoder
        'return_u_mat': encoder.return_heatmap,
        'umat_embed_layers': encoder.umat_embed_layers if hasattr(encoder, 'umat_embed_layers') else 3,
        'aug_graph_embed_layers': encoder.aug_graph_embed_layers if hasattr(encoder, 'aug_graph_embed_layers') else 3,
        'matnet_mix_score': False ##### This is hard coded for now, should be changed from the encoder
    }
    return embedder_params

def get_encoder_parameters_gcn(encoder):
    # Extract parameters necessary for initializing ResidualGatedGCNModel
    embedder_params = {
        'embed_dim': encoder.embedding_dim,
        'n_layers': encoder.n_encode_layers,
        'non_Euc': encoder.non_Euc,
        'dim_Sk': encoder.rank_k_approx * (not encoder.mul_sigma_uv),
        'rescale_dist': encoder.rescale_dist,
        'return_u_mat': encoder.return_heatmap,
        'umat_embed_layers': encoder.umat_embed_layers if hasattr(encoder, 'umat_embed_layers') else 3,
        'aug_graph_embed_layers': encoder.aug_graph_embed_layers if hasattr(encoder, 'aug_graph_embed_layers') else 3,
        'gcn_aggregation': encoder.gcn_aggregation if hasattr(encoder, 'gcn_aggregation') else "sum"
    }
    return embedder_params

# ########################################################
#### Change the forward function of embedder ###########
class embedder_added_layers_gat(GraphAttentionEncoder):
    def __init__(self, n_heads, embed_dim, n_layers, non_Euc=False, dim_Sk=0, n_edge_encode_layers=0,
                 rescale_dist=False, aug_graph_embed_layers=3, node_dim=None, normalization='batch',
                 feed_forward_hidden=512, return_u_mat=False, umat_embed_layers=3, matnet_mix_score=False):
        super().__init__(n_heads=n_heads, embed_dim=embed_dim, n_layers=n_layers, non_Euc=non_Euc, dim_Sk=dim_Sk,
                         n_edge_encode_layers=n_edge_encode_layers, rescale_dist=rescale_dist,
                         aug_graph_embed_layers=aug_graph_embed_layers, node_dim=node_dim,
                         normalization=normalization, feed_forward_hidden=feed_forward_hidden,
                         return_u_mat=return_u_mat, umat_embed_layers=umat_embed_layers,
                         matnet_mix_score=matnet_mix_score)
        
        embedding_dim = embed_dim  # Use embed_dim from the parameters

        # Parameters for the new residual layer
        self.new_weight_1 = nn.Parameter(torch.Tensor(embedding_dim, embedding_dim),requires_grad=True)
        self.new_bias_1 = nn.Parameter(torch.Tensor(embedding_dim),requires_grad=True)
        self.new_weight_2 = nn.Parameter(torch.zeros(embedding_dim, embedding_dim),requires_grad=True)
        self.new_bias_2 = nn.Parameter(torch.zeros(embedding_dim),requires_grad=True)
        torch.nn.init.xavier_uniform_(self.new_weight_1)
        torch.nn.init.constant_(self.new_bias_1, 0)
    
    def forward(self, x, S, scale_factors=None, mask=None, edge_matrix=None):

        assert mask is None, "TODO mask not yet supported!"
        
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
            # Let the embedding h go through the new residual layer
            # Pass the embbings through the new residual layer
            embeddings_eas = torch.matmul(h, self.new_weight_1) + self.new_bias_1
            # Pass the embeddings through a ReLU activation
            embeddings_eas = F.relu(embeddings_eas)
            # Pass the embeddings through the second residual layer
            embeddings_eas = torch.matmul(embeddings_eas, self.new_weight_2) + self.new_bias_2

            # The new embeddings are the sum of the original embeddings and the new embeddings
            h = h + embeddings_eas
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
            embedding_dim = h.shape[-1]**2
            # Flatten the last two dimensions
            h_flattened = h.view(h.size(0), h.size(1), -1)
            # Parameters for the new residual layer
            
            self.new_weight_1 = nn.Parameter(torch.Tensor(embedding_dim, embedding_dim).to(h.device), requires_grad=True)
            self.new_bias_1 = nn.Parameter(torch.Tensor(embedding_dim).to(h.device), requires_grad=True)
            self.new_weight_2 = nn.Parameter(torch.zeros(embedding_dim, embedding_dim).to(h.device), requires_grad=True)
            self.new_bias_2 = nn.Parameter(torch.zeros(embedding_dim).to(h.device), requires_grad=True)
            torch.nn.init.xavier_uniform_(self.new_weight_1)
            torch.nn.init.constant_(self.new_bias_1, 0)

            # print the device of h_flattened, self.new_weight_1, self.new_bias_1
            print("***", h_flattened.device, self.new_weight_1.device, self.new_bias_1.device)

            embeddings_eas = torch.matmul(h_flattened, self.new_weight_1) + self.new_bias_1
            # Pass the embeddings through a ReLU activation
            embeddings_eas = F.relu(embeddings_eas)
            # Pass the embeddings through the second residual layer
            embeddings_eas = torch.matmul(embeddings_eas, self.new_weight_2) + self.new_bias_2

            # The new embeddings are the sum of the original embeddings and the new embeddings
            h = h + embeddings_eas.view(h.shape)

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

class embedder_added_layers_gcn(ResidualGatedGCNModel):
    def __init__(self, embed_dim, n_layers, non_Euc=False, dim_Sk=0, rescale_dist=False,
                 aug_graph_embed_layers=3, return_u_mat=False, umat_embed_layers=3,
                 gcn_aggregation="sum"):
        super().__init__(embed_dim=embed_dim, n_layers=n_layers, non_Euc=non_Euc, dim_Sk=dim_Sk,
                         rescale_dist=rescale_dist, aug_graph_embed_layers=aug_graph_embed_layers,
                         return_u_mat=return_u_mat, umat_embed_layers=umat_embed_layers,
                         gcn_aggregation=gcn_aggregation)
        embedding_dim = embed_dim  # Use embed_dim from the parameters
        # Parameters for the new residual layer
        self.new_weight_1 = nn.Parameter(torch.Tensor(embedding_dim, embedding_dim),requires_grad=True)
        self.new_bias_1 = nn.Parameter(torch.Tensor(embedding_dim),requires_grad=True)
        self.new_weight_2 = nn.Parameter(torch.zeros(embedding_dim, embedding_dim),requires_grad=True)
        self.new_bias_2 = nn.Parameter(torch.zeros(embedding_dim),requires_grad=True)
        torch.nn.init.xavier_uniform_(self.new_weight_1)
        torch.nn.init.constant_(self.new_bias_1, 0)

    def forward(self, x, e, S, scale_factors=None, mask=None, adj_idx=None):
        B, V, H = x.size()
        _, _, K, _ = e.size()


        x = x.permute(0, 2, 1) # B x H x V
        e = e.permute(0, 3, 1, 2) # B x H x V x K (K = self.kNN)
        
        # GCN layers
        for l in range(len(self.layers)):
            # logger.debug(f"Layer {l}: x={x.size()}, e={e.size()}")
            x, e = self.layers[l](x, e, adj_idx)  # B x V x H, B x V x V x H
        
        if not self.return_u_mat:
            h = x.permute(0,2,1) # node features
            assert h.size() == (B, V, H), f"{h.size()} != [{B}, {V}, {H}]"
        else:
            h = e.permute(0,2,3,1)
            assert h.size() == (B, V, K, H), f"{h.size()} != [{B}, {V}, {K}, {H}]"
        
        if self.add_graph_dim > 0:
            S = S if S is not None else torch.zeros(h.size(0), 0, device=h.device)
            scale_factors = scale_factors if self.rescale_dist else torch.zeros(h.size(0), 0, device=h.device)

        # Let the embedding h go through the new residual layer
        # Pass the embbings through the new residual layer
        embeddings_eas = torch.matmul(h, self.new_weight_1) + self.new_bias_1
        # Pass the embeddings through a ReLU activation
        embeddings_eas = F.relu(embeddings_eas)
        # Pass the embeddings through the second residual layer
        embeddings_eas = torch.matmul(embeddings_eas, self.new_weight_2) + self.new_bias_2

        # The new embeddings are the sum of the original embeddings and the new embeddings
        h = h + embeddings_eas

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
            I, N, K, embed_dim = u_mat.size()
            # (I, N, N, n_heads) -> (I, N, N, n_heads+add_graph_dim)
            if self.add_graph_dim > 0:
                graph_features = torch.cat([S, scale_factors], dim=1)
                # broadcast graph features to the same shape as u_mat
                graph_features = graph_features.unsqueeze(1).unsqueeze(1).expand(I, N, N, self.add_graph_dim)
                
                u_mat = torch.cat([u_mat, graph_features], dim=3)

            u_mat = self.u_mat_embed(u_mat.view(-1, u_mat.size(-1))).view(*u_mat.size()[:3], -1).squeeze(-1)
            assert u_mat.size() == (I,N,K), f"{u_mat.size()} != [{I}, {N}, {K}]"
            u_mat = recover_full_edge_mat(u_mat, adj_idx)
            return {"heatmap": u_mat}


def replace_encoder(original_encoder, state, encoder_name, embedder_params):
    """Function to add layers to pretrained model while retaining weights from other layers."""
    if encoder_name == 'gat':
        original_encoder.embedder = embedder_added_layers_gat(**embedder_params)
    if encoder_name == 'gcn':
        original_encoder.embedder = embedder_added_layers_gcn(**embedder_params)
        # original_encoder.embedder.forward = embedder_added_layers_gcn(original_encoder, embedder_params['embed_dim']) # BUG: what are you doing?!
    original_encoder.embedder.load_state_dict(state_dict=state, strict=False)
    return original_encoder

#######################################################################################################################
# EAS Training process
#######################################################################################################################

def run_eas_lay_encoder(encoder, grouped_actor, instance_data, encoder_name, problem_name, eval_opts, max_runtime=1000):

    # EAS-Lay parameters
    ACTOR_WEIGHT_DECAY = 1e-6
    param_lr = 0.0032 # EAS Learning rate
    p_runs = 1  # Number of parallel runs per instance, set 1 here
    max_iter = 2000 # Maximum number of EAS iterations
    param_lambda = 0.05 # Imitation learning loss weight
    # max_runtime = 1000 # Maximum runtime in seconds




    # Save the original state_dict of the encoder (e.p. the parameters of the encoder)
    original_embedder_state_dict = encoder.embedder.state_dict()
    if encoder_name == 'gat':
        embedder_params = get_encoder_parameters_gat(encoder)
    if encoder_name == 'gcn':
        embedder_params = get_encoder_parameters_gcn(encoder)

    # Load the instances
    ###############################################
    with torch.no_grad():
    # Didn't do Augment here because it's not necessary for the Non-eculidean case
        if use_cuda:
            encoder_modified = replace_encoder(encoder, original_embedder_state_dict, encoder_name,
                                                     embedder_params).cuda()
        else:
            print("No GPU available")
            encoder_modified = replace_encoder(encoder, original_embedder_state_dict, encoder_name,
                                                     embedder_params)

    # Only update the weights of the added layer during training
    optimizer = optim.Adam(
        [encoder_modified.embedder.new_weight_1,
        encoder_modified.embedder.new_bias_1,
        encoder_modified.embedder.new_weight_2,
        encoder_modified.embedder.new_bias_2], lr=param_lr,
        weight_decay=ACTOR_WEIGHT_DECAY)


    # Start the search
    ###############################################
    t_start = time.time()
    for iter in range(max_iter):
        embed = encoder_modified(instance_data)
        batch_rep=1
        iter_rep=1
        input, embed = do_batch_rep((instance_data, embed), batch_rep)

        costs = []
        _log_p_s = []
        for i in range(iter_rep):
            cost, _log_p, pi = grouped_actor(input, return_pi=True, **embed)

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
    return encoder_modified