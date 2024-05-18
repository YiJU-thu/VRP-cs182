from torch.utils.data import Dataset
import torch
import os, sys
import pickle

from problems.vrp.state_cvrp import StateCVRP
from problems.vrp.state_sdvrp import StateSDVRP
from utils.beam_search import beam_search

from loguru import logger


curr_path = os.path.dirname(__file__)
utils_vrp_path = os.path.join(curr_path, '..', '..', '..', 'utils_project')
if utils_vrp_path not in sys.path:
    sys.path.append(utils_vrp_path)
from utils_vrp import get_random_graph, normalize_graph, recover_graph,\
      get_tour_len_torch, to_torch, get_rel_dist_mat_batch


class CVRP(object):

    NAME = 'cvrp'  # Capacitated Vehicle Routing Problem

    VEHICLE_CAPACITY = 1.0  # (w.l.o.g. vehicle capacity is 1, demands should be scaled)

    @staticmethod
    def get_costs(dataset, pi):
        batch_size, graph_size = dataset['demand'].size()
        # Check that tours are valid, i.e. contain 0 to n -1
        sorted_pi = pi.data.sort(1)[0]

        # Sorting it should give all zeros at front and then 1...n
        assert (
            torch.arange(1, graph_size + 1, out=pi.data.new()).view(1, -1).expand(batch_size, graph_size) ==
            sorted_pi[:, -graph_size:]
        ).all() and (sorted_pi[:, :-graph_size] == 0).all(), "Invalid tour"

        # Visiting depot resets capacity so we add demand = -capacity (we make sure it does not become negative)
        demand_with_depot = torch.cat(
            (
                torch.full_like(dataset['demand'][:, :1], -CVRP.VEHICLE_CAPACITY),
                dataset['demand']
            ),
            1
        )
        d = demand_with_depot.gather(1, pi)

        used_cap = torch.zeros_like(dataset['demand'][:, 0])
        for i in range(pi.size(1)):
            used_cap += d[:, i]  # This will reset/make capacity negative if i == 0, e.g. depot visited
            # Cannot use less than 0
            used_cap[used_cap < 0] = 0
            assert (used_cap <= CVRP.VEHICLE_CAPACITY + 1e-5).all(), "Used more than capacity"

        # # Gather dataset in order of tour
        # loc_with_depot = dataset['coords']
        # d = loc_with_depot.gather(1, pi[..., None].expand(*pi.size(), loc_with_depot.size(-1)))

        # # Length is distance (L2-norm of difference) of each next location to its prev and of first and last to depot
        # return (
        #     (d[:, 1:] - d[:, :-1]).norm(p=2, dim=2).sum(1)
        #     + (d[:, 0] - dataset['coords'][:,0]).norm(p=2, dim=1)  # Depot to first
        #     + (d[:, -1] - dataset['coords'][:,0]).norm(p=2, dim=1)  # Last to depot, will be 0 if depot is last
        # ), None
    
        return get_tour_len_torch(dataset, pi), None

    @staticmethod
    def make_dataset(*args, **kwargs):
        return VRPDataset(*args, **kwargs)

    @staticmethod
    def make_state(*args, **kwargs):
        return StateCVRP.initialize(*args, **kwargs)

    @staticmethod
    def beam_search(input, beam_size, expand_size=None,
                    compress_mask=False, model=None, max_calc_batch_size=4096):

        assert model is not None, "Provide model"

        fixed = model.precompute_fixed(input)

        def propose_expansions(beam):
            return model.propose_expansions(
                beam, fixed, expand_size, normalize=True, max_calc_batch_size=max_calc_batch_size
            )

        state = CVRP.make_state(
            input, visited_dtype=torch.int64 if compress_mask else torch.uint8
        )

        return beam_search(state, beam_size, propose_expansions)

# We dont need this class for now
class SDVRP(object):

    NAME = 'sdvrp'  # Split Delivery Vehicle Routing Problem

    VEHICLE_CAPACITY = 1.0  # (w.l.o.g. vehicle capacity is 1, demands should be scaled)

    @staticmethod
    def get_costs(dataset, pi):
        batch_size, graph_size = dataset['demand'].size()

        # Each node can be visited multiple times, but we always deliver as much demand as possible
        # We check that at the end all demand has been satisfied
        demands = torch.cat(
            (
                torch.full_like(dataset['demand'][:, :1], -SDVRP.VEHICLE_CAPACITY),
                dataset['demand']
            ),
            1
        )
        rng = torch.arange(batch_size, out=demands.data.new().long())
        used_cap = torch.zeros_like(dataset['demand'][:, 0])
        a_prev = None
        for a in pi.transpose(0, 1):
            assert a_prev is None or (demands[((a_prev == 0) & (a == 0)), :] == 0).all(), \
                "Cannot visit depot twice if any nonzero demand"
            d = torch.min(demands[rng, a], SDVRP.VEHICLE_CAPACITY - used_cap)
            demands[rng, a] -= d
            used_cap += d
            used_cap[a == 0] = 0
            a_prev = a
        assert (demands == 0).all(), "All demand must be satisfied"

        # Gather dataset in order of tour
        loc_with_depot = torch.cat((dataset['depot'][:, None, :], dataset['loc']), 1)
        d = loc_with_depot.gather(1, pi[..., None].expand(*pi.size(), loc_with_depot.size(-1)))

        # Length is distance (L2-norm of difference) of each next location to its prev and of first and last to depot
        return (
            (d[:, 1:] - d[:, :-1]).norm(p=2, dim=2).sum(1)
            + (d[:, 0] - dataset['depot']).norm(p=2, dim=1)  # Depot to first
            + (d[:, -1] - dataset['depot']).norm(p=2, dim=1)  # Last to depot, will be 0 if depot is last
        ), None

    @staticmethod
    def make_dataset(*args, **kwargs):
        return VRPDataset(*args, **kwargs)

    @staticmethod
    def make_state(*args, **kwargs):
        return StateSDVRP.initialize(*args, **kwargs)

    @staticmethod
    def beam_search(input, beam_size, expand_size=None,
                    compress_mask=False, model=None, max_calc_batch_size=4096):
        assert model is not None, "Provide model"
        assert not compress_mask, "SDVRP does not support compression of the mask"

        fixed = model.precompute_fixed(input)

        def propose_expansions(beam):
            return model.propose_expansions(
                beam, fixed, expand_size, normalize=True, max_calc_batch_size=max_calc_batch_size
            )

        state = SDVRP.make_state(input)

        return beam_search(state, beam_size, propose_expansions)

# We don't need this function for now
def make_instance(args):
    depot, loc, demand, capacity, *args = args
    grid_size = 1
    if len(args) > 0:
        depot_types, customer_types, grid_size = args
    return {
        'loc': torch.tensor(loc, dtype=torch.float) / grid_size,
        'demand': torch.tensor(demand, dtype=torch.float) / capacity,
        'depot': torch.tensor(depot, dtype=torch.float) / grid_size
    }


class VRPDataset(Dataset):
    
    def __init__(self, filename=None, dataset=None, size=50, num_samples=1000000, offset=0, 
                 non_Euc=False, rand_dist="standard", rescale=False, distribution=None, force_triangle_iter=2, no_coords=False, keep_rel=False,
                 normalize_loaded=True):
        
        super(VRPDataset, self).__init__()
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
            self.data = get_random_graph(n=size, num_graphs=num_samples, non_Euc=non_Euc, rescale=rescale_tmp, 
                                         force_triangle_iter=force_triangle_iter, no_coords=no_coords, keep_rel=keep_rel, is_cvrp=True)
            if (not rescale) and rescale_tmp:
                self.data = recover_graph(self.data)

        # FIXME: this may be wired
        if no_coords:
            self.data["distance"] /= 1e5    # entries are originally in [1,1e6], this makes the optimal tour length around 15.7


        # assert self.data.get("coords", self.data.get("distance")).device == torch.device("cpu"), "Data should be on CPU"
        if not keep_rel and "rel_distance" in self.data:
            del self.data["rel_distance"]
        if keep_rel and "distance" in self.data and "rel_distance" not in self.data:
            assert "coords" in self.data, "Need coords to compute rel_distance"
            assert self.data.get("scale_factors") is None, "FIXME: scale_factors not supported with rel_distance"
            coords = self.data["coords"]
            dist_mat = self.data["distance"]
            self.data["rel_distance"] = get_rel_dist_mat_batch(coords, dist_mat)

    @property
    def size(self):
        if "coords" in self.data:
            return self.data["coords"].shape[0]
        elif "distance" in self.data:
            return self.data["distance"].shape[0]
        else:
            raise NotImplementedError("data has no 'coords' or 'distance' key")

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # return self.data[idx]
        # note: DataParallel requires everything does not support None type
        scale_factors = torch.tensor([float('nan')]) if not self.rescale else self.data['scale_factors'][idx]
        # logger.debug(self.data.keys())
        data = {}
        for k, v in self.data.items():
            if k in ["coords", "distance", "rel_distance", "demand"]:
                data[k] = v[idx]
            elif k == "scale_factors":
                data[k] = scale_factors
        return data

    def pomo_augment(self, N1, N2):
        raise NotImplementedError("This function is not implemented yet")
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
        if "coords" in dataset:
            dataset['coords'] = sample_coords_start(dataset['coords'], N1)
        if self.non_Euc:
            dataset['distance'] = sample_dist_mat_start(dataset['distance'], N1)
            if "rel_distance" in dataset:
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
        if "coords" in dataset:
            dataset['coords'] = sample_coords_rot(dataset['coords'], N2)
        if self.non_Euc:
            dataset['distance'] = sample_dist_mat_rot(dataset['distance'], N2)
            if "rel_distance" in dataset:
                dataset['rel_distance'] = sample_dist_mat_rot(dataset['rel_distance'], N2)
        if dataset["scale_factors"] is not None:
            dataset["scale_factors"] = repeat_scale_factors(dataset["scale_factors"], N2)

        self.data = dataset