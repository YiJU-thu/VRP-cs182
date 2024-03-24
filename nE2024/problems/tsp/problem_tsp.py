from torch.utils.data import Dataset
import torch
import os, sys
import pickle
from problems.tsp.state_tsp import StateTSP
from utils.beam_search import beam_search
from loguru import logger

curr_path = os.path.dirname(__file__)
utils_vrp_path = os.path.join(curr_path, '..', '..', '..', 'utils_project')
if utils_vrp_path not in sys.path:
    sys.path.append(utils_vrp_path)
from utils_vrp import get_random_graph, normalize_graph, recover_graph,\
      get_tour_len_torch, to_torch

class TSP(object):

    NAME = 'tsp'

    @staticmethod
    def get_costs(dataset, pi):
        # Check that tours are valid, i.e. contain 0 to n -1
        assert (
            torch.arange(pi.size(1), out=pi.data.new()).view(1, -1).expand_as(pi) ==
            pi.data.sort(1)[0]
        ).all(), "Invalid tour"

        # # Gather dataset in order of tour
        # d = dataset.gather(1, pi.unsqueeze(-1).expand_as(dataset))

        # # FIXME: don't fully understand API - should dictionary be passed in?
        # # use utils_vrp/recover_graph to recover true distance matrix here

        # # Length is distance (L2-norm of difference) from each next location from its prev and of last from first
        # return (d[:, 1:] - d[:, :-1]).norm(p=2, dim=2).sum(1) + (d[:, 0] - d[:, -1]).norm(p=2, dim=1), None
        return get_tour_len_torch(dataset, pi), None


    @staticmethod
    def make_dataset(*args, **kwargs):
        return TSPDataset(*args, **kwargs)

    @staticmethod
    def make_state(*args, **kwargs):
        return StateTSP.initialize(*args, **kwargs)

    @staticmethod
    def beam_search(input, beam_size, expand_size=None,
                    compress_mask=False, model=None, max_calc_batch_size=4096):

        assert model is not None, "Provide model"

        fixed = model.precompute_fixed(input)

        def propose_expansions(beam):
            return model.propose_expansions(
                beam, fixed, expand_size, normalize=True, max_calc_batch_size=max_calc_batch_size
            )

        state = TSP.make_state(
            input, visited_dtype=torch.int64 if compress_mask else torch.uint8
        )

        return beam_search(state, beam_size, propose_expansions)


class TSPDataset(Dataset):
    
    def __init__(self, filename=None, dataset=None, size=50, num_samples=1000000, offset=0, 
                 non_Euc=False, rand_dist="standard", rescale=False, distribution=None, force_triangle_iter=2,
                 normalize_loaded=True):
        super(TSPDataset, self).__init__()
        self.non_Euc = non_Euc
        self.rescale = rescale

        self.data_set = []
        if filename is not None or dataset is not None:
            if filename is not None:
                assert dataset is None
                assert filename.endswith('.pkl')
                # assert os.path.splitext(filename)[1] == '.pkl'

                with open(filename, 'rb') as f:
                    data = pickle.load(f)
            else:
                data = dataset
            
            if isinstance(data, dict):
                # keys are: coords, distance, (rel_distance, scale_factors)
                data = to_torch(data, device="cpu")
                if normalize_loaded:
                    data = normalize_graph(data, rescale=rescale)
                self.data = data

            else:
                # TODO: old version, inputs would be (I,N,2) ndarray
                raise NotImplementedError



        else:
            # Sample points randomly in [0, 1] square
            # self.data = [torch.FloatTensor(size, 2).uniform_(0, 1) for i in range(num_samples)]
            assert rand_dist in ["standard", "complex"]
            if rand_dist == "standard":
                assert rescale == False
            rescale_tmp = (rand_dist == "complex")
            self.data = get_random_graph(n=size, num_graphs=num_samples, non_Euc=non_Euc, rescale=rescale_tmp, force_triangle_iter=force_triangle_iter)
            if (not rescale) and rescale_tmp:
                self.data = recover_graph(self.data)


        # self.size = self.data["coords"].shape[0]

    @property
    def size(self):
        return self.data["coords"].shape[0]

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # note: DataParallel requires everything does not support None type
        scale_factors = torch.tensor([float('nan')]) if not self.rescale else self.data['scale_factors'][idx]
        if not self.non_Euc:
            return {
                "coords": self.data['coords'][idx],
                "scale_factors": scale_factors,
            }
        else:
            return {
                "coords": self.data['coords'][idx],
                "distance": self.data['distance'][idx],
                "rel_distance": self.data['rel_distance'][idx],
                "scale_factors": scale_factors,
            }
    
    def pomo_augment(self, N1, N2):
        dataset = self.data

        def sample_coords_start(x, N1):
            if N1 == 1 or N1 is None:
                return x
            B, N, _ = x.shape   # shape: (batch_size, graph_size, 2)

            # generate N1 batches, batch i is the original batch shifted by -i (so the first node is the -i th node)
            # transpose the first two dimensions so that instances from the same original instances are together
            x_augment = torch.stack([torch.roll(x, shifts= -i%N, dims=1) for i in range(N1)]).\
                transpose(0, 1).\
                    reshape(B*N1, N, 2)
            assert x_augment.shape == (B*N1, N, 2)
            assert torch.norm(x_augment[0, 0] - x_augment[1, -1]) < 1e-8  # graph 0's first node is graph 1's last node
            return x_augment
        
        def sample_dist_mat_start(x, N1):
            if N1 == 1 or N1 is None:
                return x
            B, N, _ = x.shape   # shape: (batch_size, graph_size, graph_size)
            
            # generate N1 batches, batch i is the original batch shifted by -i in both row & column
            # (so the first node is the -i th node)
            # transpose the first two dimensions so that instances from the same original instances are together
            x_augment = torch.stack([torch.roll(x, shifts= (-i%N,-i%N), dims=(1,2)) for i in range(N1)]).\
                transpose(0, 1).\
                    reshape(B*N1, N, N)
            assert x_augment.shape == (B*N1, N, N)
            assert torch.norm(x_augment[0, 0, 1] - x_augment[1, -1, 0]) < 1e-8  # graph 0's first node is graph 1's last node
            return x_augment
        
        def repeat_scale_factors(x, N1):
            if N1 == 1 or N1 is None:
                return x
            B, d = x.shape
            x_augment = x[:,None,:].repeat(1, N1, 1).reshape(B*N1, d)
            return x_augment


        # sample N1 starts for each instance
        dataset['coords'] = sample_coords_start(dataset['coords'], N1)
        if self.non_Euc:
            dataset['distance'] = sample_dist_mat_start(dataset['distance'], N1)
            dataset['rel_distance'] = sample_dist_mat_start(dataset['rel_distance'], N1)
        if dataset["scale_factors"] is not None:
            dataset["scale_factors"] = repeat_scale_factors(dataset["scale_factors"], N1)
        

        def sample_coords_rot(x, N2):
            if N2==1 or N2 is None:
                return x
            B_N1, N, _ = x.shape  # shape: (batch_size*N1, graph_size, 2)
            
            # generate N2 rotation matrices
            rot_mats = [torch.tensor([[torch.cos(theta), -torch.sin(theta)],
                                                [torch.sin(theta), torch.cos(theta)]])
                        for theta in torch.rand(N2) * 2 * torch.pi]
            assert len(rot_mats) == N2
            
            # Rotate the matrices around (0.5, 0.5): 
            # first translate the matrix to center at (0, 0), then rotate, then translate back
            # (the origin coords are in a unit square, so the center is (0.5, 0.5))
            x_augment = torch.stack([torch.matmul(x - 0.5, rot_mat) + 0.5 for rot_mat in rot_mats]).\
                transpose(0, 1).\
                    reshape(B_N1*N2, N, 2)
            assert x_augment.shape == (B_N1*N2, N, 2)
            return x_augment
        
        def sample_dist_mat_rot(x, N2):
            if N2==1 or N2 is None:
                return x
            B_N1, N, _ = x.shape # shape: (batch_size*N1, graph_size, graph_size)
            x_augment = torch.stack([x for _ in range(N2)]).transpose(0, 1).reshape(B_N1*N2, N, N)
            return x_augment

        # sample N2 rotations for each instance
        dataset['coords'] = sample_coords_rot(dataset['coords'], N2)
        if self.non_Euc:
            dataset['distance'] = sample_dist_mat_rot(dataset['distance'], N2)
            dataset['rel_distance'] = sample_dist_mat_rot(dataset['rel_distance'], N2)
        if dataset["scale_factors"] is not None:
            dataset["scale_factors"] = repeat_scale_factors(dataset["scale_factors"], N2)

        self.data = dataset
