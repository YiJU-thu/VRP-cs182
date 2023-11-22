from torch.utils.data import Dataset
import torch
import os, sys
import pickle
from problems.tsp.state_tsp import StateTSP
from utils.beam_search import beam_search

curr_path = os.path.dirname(__file__)
utils_vrp_path = os.path.join(curr_path, '..', '..', '..', 'utils_project')
if utils_vrp_path not in sys.path:
    sys.path.append(utils_vrp_path)
from utils_vrp import get_random_graph, normalize_graph, get_tour_len_torch

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
    
    def __init__(self, filename=None, size=50, num_samples=1000000, offset=0, non_Euc=False, rescale=False, distribution=None):
        super(TSPDataset, self).__init__()
        self.non_Euc = non_Euc
        self.rescale = rescale

        self.data_set = []
        if filename is not None:
            raise NotImplementedError
            assert os.path.splitext(filename)[1] == '.pkl'

            with open(filename, 'rb') as f:
                data = pickle.load(f)
                self.data = [torch.FloatTensor(row) for row in (data[offset:offset+num_samples])]
        else:
            # Sample points randomly in [0, 1] square
            # self.data = [torch.FloatTensor(size, 2).uniform_(0, 1) for i in range(num_samples)]
            self.data = get_random_graph(n=size, num_graphs=num_samples, non_Euc=non_Euc, rescale=rescale)

        self.size = self.data["coords"].shape[0]

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
