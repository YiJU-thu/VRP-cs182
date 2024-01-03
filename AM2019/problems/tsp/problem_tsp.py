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
                 non_Euc=False, rand_dist="standard", rescale=False, distribution=None, force_triangle_iter=2):
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
                data = to_torch(data)
                self.data = normalize_graph(data, rescale=rescale)

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
            if not rescale:
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

        def sample_coordinate_for_batch(batch_coordinate_data, N1):
            B, N, _ = batch_coordinate_data.shape
            # batch_coordinate_data with shape (batch_size, N, 2)
            transposed_matrices = torch.stack([torch.roll(batch_coordinate_data, shifts= -i%N, dims=1) for i in range(N1)])

            # Transpose the first two dimensions
            transposed_matrices = transposed_matrices.transpose(0, 1)

            # Reshape to combine the first two dimensions
            new_matrix = transposed_matrices.reshape(B*N1, N, 2)
            return new_matrix
        
        def reorder_distance_for_batch(batch_distance_data, N1):
            B, N, _ = batch_distance_data.shape
            transposed_matrices = torch.stack([torch.roll(batch_distance_data, shifts= (-i%N,-i%N), dims=(1,2)) for i in range(N1)])
            # Transpose the first two dimensions
            transposed_matrices = transposed_matrices.transpose(0, 1)
            # Reshape the matrix to shape (B*N1, N, N)
            combined_matrix = transposed_matrices.reshape(B*N1, N, N)
            return combined_matrix
        
        if N1 is not None:
            # sample N1 starts for each instance
            dataset['coords'] = sample_coordinate_for_batch(dataset['coords'], N1)
            if self.non_Euc:
                dataset['distance'] = reorder_distance_for_batch(dataset['distance'], N1)
                dataset['rel_distance'] = reorder_distance_for_batch(dataset['rel_distance'], N1)
        
        def rotated_coordinate_for_batch(batch_coordinate_data, N2):
            B_N1, N, _ = batch_coordinate_data.shape
            matrices = []
            for _ in range(N2):
                # Generate a random angle
                theta = torch.rand(1) * 2 * torch.pi

                # Create the rotation matrix
                rotation_matrix = torch.tensor([[torch.cos(theta), -torch.sin(theta)],
                                                [torch.sin(theta), torch.cos(theta)]])
                matrices.append(rotation_matrix)
            # Rotate the matrices around (0.5, 0.5)
            matrix_subtracted = batch_coordinate_data - 0.5
            rotated_matrices = torch.stack([torch.matmul(matrix_subtracted, rotation_matrix) for rotation_matrix in matrices])
            rotated_matrices += 0.5
            rotated_matrices = (rotated_matrices.transpose(0, 1)).reshape(B_N1*N2, N, 2)
            return rotated_matrices
        
        def copy_distance_for_batch(batch_distance_data, N2):
            B_N1, N, _ = batch_distance_data.shape
            copied_matrices = torch.stack([batch_distance_data for _ in range(N2)])
            copied_matrices = (copied_matrices.transpose(0, 1)).reshape(B_N1*N2, N, N)
            return copied_matrices

        if N2 is not None:
            # sample N2 rotations for each instance
            dataset['coords'] = rotated_coordinate_for_batch(dataset['coords'], N2)
            if self.non_Euc:
                dataset['distance'] = copy_distance_for_batch(dataset['distance'], N2)
                dataset['rel_distance'] = copy_distance_for_batch(dataset['rel_distance'], N2)

        # update the dataset，dataset is a dictionary
        # dataset['coords']: (B*N1*N2, N, 2)
        # dataset['distance']: (B*N1*N2, N, N)
        
        self.data = dataset
