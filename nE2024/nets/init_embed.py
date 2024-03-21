import torch
from torch import nn

from utils.tensor_functions import randomized_svd_batch


from loguru import logger


class InitEncoder(nn.Module):

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
                 ):
        super(InitEncoder, self).__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        self.allow_partial = problem.NAME == 'sdvrp'
        self.is_vrp = problem.NAME == 'cvrp' or problem.NAME == 'sdvrp'
        self.is_orienteering = problem.NAME == 'op'
        self.is_pctsp = problem.NAME == 'pctsp'

        self.problem = problem

        self.non_Euc = non_Euc
        # idea 1: node feature augmentation
        self.rank_k_approx = rank_k_approx
        self.svd_original_edge = svd_original_edge
        self.mul_sigma_uv = mul_sigma_uv
        self.full_svd = full_svd
        self.only_distance = only_distance

        # assert problem.NAME == 'tsp', "Only tsp is supported at the moment"
        # if only_distance:
        if only_distance:
            assert non_Euc == True, "only_distance is only supported for non-Euclidean input"
            assert rank_k_approx > 0, "only_distance is not supported for rank_k_approx = 0"
            assert svd_original_edge == True, "must svd on the original edge matrix if only_distance is True"

        node_dim = 2*(1-only_distance) + 2 * rank_k_approx  # x, y, u_i_k, v_i_k
        # Problem specific context parameters (placeholder and step context dimension)
        if self.is_vrp or self.is_orienteering or self.is_pctsp:
            # Embedding of last node + remaining_capacity / remaining length / remaining prize to collect

            if self.is_pctsp:
                feature_dim = 2 # expected_prize, penalty
            else:
                feature_dim = 1  # demand / prize
            node_dim += feature_dim

            # Special embedding projection for depot node
            self.init_embed_depot = nn.Linear(node_dim-feature_dim, embedding_dim)
            
            # We dont do split delivery now
            if self.is_vrp and self.allow_partial:  # Need to include the demand if split delivery allowed
                self.project_node_step = nn.Linear(1, 3 * embedding_dim, bias=False)
        
        else:  # TSP
            assert problem.NAME == "tsp", "Unsupported problem: {}".format(problem.NAME)

        self.init_embed = nn.Linear(node_dim, embedding_dim)


    def forward(self, input):
        """
        :param input: (batch_size, graph_size, node_dim) input node features or dictionary with multiple tensors
        :return:
        """

        # return init_embed, Sk

        return self.get_init_embed(input)


    def get_init_embed(self, input):
        
        # add sanity check for input
        assert isinstance(input, dict), "Input must be a dictionary"
        assert 'coords' in input, "Input must contain 'coords' key"
        I, N, _ = input['coords'].shape
        if not self.non_Euc: # input is Euclidean
            assert self.rank_k_approx == 0, "rank_k_approx is not supported for Euclidean input"
            assert self.only_distance == False, "only_distance is not supported for Euclidean input"
            scale_factors_dim = 1
        else: # non-Euclidean
            assert "rel_distance" in input, "Input must contain 'rel_distance' key"
            assert input["distance"].shape == (I, N, N), "distance must be of shape (I, N, N)"
            assert input["rel_distance"].shape == (I, N, N), "rel_distance must be of shape (I, N, N)"
            scale_factors_dim = 3

        # ================================================

        return self._init_embed(input) # (init_embed, Sk)


    def _init_embed(self, input):        
        # svd and add node features, then go through the linear layer
        coords = input['coords']
        I, N, _ = coords.shape
        if self.rank_k_approx == 0:
            nodes = coords
            Sk = None
        else:
            mat_to_svd = input['distance'] if self.svd_original_edge else input['rel_distance']

            if self.full_svd or self.rank_k_approx/N > 0.6:
                U, S, Vh = torch.linalg.svd(mat_to_svd)
                V = Vh.mH
                Uk, Sk, Vk = U[..., :self.rank_k_approx], S[..., :self.rank_k_approx], V[..., :self.rank_k_approx]
            else:
                try: # avoid ill-conditioned matrix, not converge, etc...
                    Uk, Sk, Vk = randomized_svd_batch(mat_to_svd, self.rank_k_approx)
                except Exception as e:
                    logger.warning("randomized_svd_batch failed, using full svd instead")
                    logger.warning(e)
                    U, S, Vh = torch.linalg.svd(mat_to_svd)
                    V = Vh.mH
                    Uk, Sk, Vk = U[..., :self.rank_k_approx], S[..., :self.rank_k_approx], V[..., :self.rank_k_approx]
            
            if self.rank_k_approx > N:
                # pad Uk, Vk, Sk with zeros
                Uk = torch.cat([Uk, torch.zeros(Uk.shape[0], Uk.shape[2], self.rank_k_approx-N, device=Uk.device)], dim=2)
                Vk = torch.cat([Vk, torch.zeros(Vk.shape[0], Vk.shape[2], self.rank_k_approx-N, device=Vk.device)], dim=2)
                Sk = torch.cat([Sk, torch.zeros(Sk.shape[0], self.rank_k_approx-N, device=Sk.device)], dim=1)

             # multiply sigma to U and V
            if self.mul_sigma_uv:
                # Uk: (B, N, K), Sk: (B, K), Vk: (B, N, K)
                sqrt_S = torch.sqrt(Sk[:, None, :])
                Uk, Vk = Uk * sqrt_S, Vk * sqrt_S
                # Sk has been factored into Uk and Vk, so no gragh feature needed
                Sk = None

            if self.only_distance:
                nodes = torch.cat([Uk, Vk], dim=2)
            else:
                nodes = torch.cat([coords, Uk, Vk], dim=2)
        
        if self.is_vrp or self.is_orienteering or self.is_pctsp: # VRP, OP, PCTSP
            if self.is_vrp:
                features = ('demand', )
            elif self.is_orienteering:
                features = ('prize', )
            else:
                assert self.is_pctsp
                features = ('deterministic_prize', 'penalty')
            assert nodes.shape == (coords.shape[0], coords.shape[1], 2*(1-self.only_distance) + 2 * self.rank_k_approx)
            return torch.cat(
                (
                    self.init_embed_depot(nodes[:,0])[:, None, :],
                    self.init_embed(torch.cat((
                        nodes[:,1:],
                        *(input[feat][:, :, None] for feat in features)
                    ), -1))
                ),
                1
            ), Sk
        else: #TSP
            assert nodes.shape == (coords.shape[0], coords.shape[1], 2*(1-self.only_distance) + 2 * self.rank_k_approx), "nodes.shape is {}".format(nodes.shape)
            return self.init_embed(nodes), Sk