import numpy
import numpy as np
from loguru import logger
import torch
from torch.distributions.distribution import Distribution as Distribution
from torch.distributions.uniform import Uniform
from torch.distributions.normal import Normal
from torch.distributions.log_normal import LogNormal


def deprecated(func):
    def new_func(*args, **kwargs):
        logger.warning(f"Call to deprecated function {func.__name__}. Use get_{func.__name__} instead.")
        return func(*args, **kwargs)

    return new_func


def to_np(data):
    for key in data:
        if isinstance(data[key], torch.Tensor):
            data[key] = data[key].numpy()
    return data


def get_euclidean_dist_matrix(points):
    return torch.cdist(points, points, p=2)


def get_tour_len(tour, coords=None, dist_mat=None, norm=None):
    """
    Compute the length of a tour.

    args:
        tour: shape = (N,)
        coords: shape = (N,2)
        dist_mat: shape = (N,N)
        norm: "L1", "L2", "Linf", or None
            (1) if None, and dist_mat is given: use dist_mat
            (2) else, coords should be given, compute L2 (Euclidean) norm
    """

    if dist_mat is not None:
        assert coords is None
        assert norm is None
        if dist_mat.shape != (len(tour), len(tour)):
            logger.warning(f"dist_mat shape {dist_mat.shape} != tour len {len(tour)}")
        # assert dist_mat.shape == (len(tour), len(tour))
        return np.sum(dist_mat[tour, np.roll(tour, -1)])
    else:
        assert coords is not None
        assert coords.shape == (len(tour), 2)
        norm = "L2" if norm is None else norm
        assert norm in ["L1", "L2", "Linf"], "Invalid norm"
        v = coords[tour]
        v1 = np.roll(v, -1, axis=0)
        if norm == "L2":
            return np.sum(np.sqrt(((v1 - v) ** 2).sum(axis=1)))
        if norm == "L1":
            return np.sum(np.abs(v1 - v).sum(axis=1))
        if norm == "Linf":
            return np.sum(np.max(np.abs(v1 - v), axis=1))


def get_tour_len_batch(tour, coords=None, dist_mat=None, norm=None):
    """
    Compute the length of a batch of instances. (I=batch size)

    args:
        tour: shape = (I,N)
        coords: shape = (I,N,2)
        dist_mat: shape = (I,N,N)
        norm: "L1", "L2", "Linf", or None
            (1) if None, and dist_mat is given: use dist_mat
            (2) else, coords should be given, compute L2 (Euclidean) norm
    """

    raise NotImplementedError


def get_tour_len_torch(data, tour):
    assert isinstance(data, dict)
    if data.get("scale_factors") is not None:
        data = recover_graph(data)
    if data.get("distance") is not None:
        dist_mat = data["distance"]
    else:
        dist_mat = get_euclidean_dist_matrix(data["coords"])
    
    dist_mat = data["distance"] # shape=(I,N,N)
    (I, N, _) = data["coords"].shape
    assert tour.shape == (I, N)
    t0 = tour.flatten()
    t1 = torch.roll(tour, -1, dims=1).flatten()
    idx_flatten = torch.arange(I * N, device=tour.device) // N
    cost = dist_mat[idx_flatten, t0, t1].reshape(I, N) # shape=(I,N)
    cost = torch.sum(cost, dim=1)   # shape=(I,)


    c = torch.sum(dist_mat[0][tour[0], torch.roll(tour[0],-1)])
    assert torch.allclose(c, cost[0])
    return cost




def get_normed_dist_mat(coords, norm="L2"):
    """
    Compute the normed distance matrix (i.e., the distance matrix when the norm is "L2").
    """
    assert coords.shape[1] == 2
    assert norm in ["L1", "L2", "Linf"], "Invalid norm"
    expand = coords[:, None, :] - coords[None, :, :]
    if norm == "L2":
        norm_dist_mat = np.sqrt((expand ** 2).sum(axis=2))
    if norm == "L1":
        norm_dist_mat = np.abs(expand).sum(axis=2)
    if norm == "Linf":
        norm_dist_mat = np.max(np.abs(expand), axis=2)
    return norm_dist_mat


def get_normed_dist_mat_batch(coords, norm="L2"):
    """
    Batched version of get_normed_dist_mat, using torch.
    """
    assert norm in ["L1", "L2", "Linf"], "Invalid norm"
    if norm == "L1":
        pnorm = 1.0
    elif norm == "L2":
        pnorm = 2.0
    else:
        pnorm = float("Inf")
    norm_dist_mat = torch.cdist(coords, coords, p=pnorm)
    return norm_dist_mat


def get_rel_dist_mat(coords, dist_mat, norm="L2", fill=1):
    """
    Normalize the distance matrix relative to a simple normed distance, 
    i.e., how much the distance matrix deviates from 
        the one when the distance is a norm (e.g., "L2"), which info has effectively embedded in coords.
    
    args:
        coords (np.ndarray): shape = (N,2)
        dist_mat (np.ndarray): shape = (N,N)
        norm (str): "L1", "L2", "Linf", default = "L2"
        fill (float): fill the relative distance matrix with this value where face x/0, default = 1 (arbitrary value works)
    """

    assert coords.shape == (len(dist_mat), 2)
    assert dist_mat.shape == (len(dist_mat), len(dist_mat))

    # get the normed (default 'L2') distance matrix
    norm_dist_mat = get_normed_dist_mat(coords, norm=norm)

    dist_mat_rel = np.divide(dist_mat, norm_dist_mat, out=np.ones_like(dist_mat) * fill, where=norm_dist_mat != 0)
    # dist_mat_rel /= dist_mat_rel.mean() # normalize the mean to 1
    # instead of normalize the mean to 1, we normalize the mean of log(x) to 0
    eps = 1e-10
    dist_mat_rel /= np.exp(np.log(dist_mat_rel + eps).mean())

    return dist_mat_rel


def get_rel_dist_mat_batch(coords, dist_mat, norm="L2", fill=1):
    """
    Batched version of get_rel_dist_mat, using only torch

    args:
        coords: (I, N, 2) tensor
        dist_mat: (I, N, N) tensor
        norm (str): "L1", "L2", "Linf", default = "L2"
        fill (float): fill the relative distance matrix with this value where face x/0, default = 1 (arbitrary value works)

    returns:
        dist_mat_rel: (I, N, N) relative distance tensor
    """
    assert isinstance(coords, torch.Tensor) and isinstance(dist_mat, torch.Tensor)
    assert len(coords.shape) == 3
    (I, N, _) = coords.shape
    assert dist_mat.shape == (I, N, N)

    # get the normed (default 'L2') distance matrix
    norm_dist_mat = get_normed_dist_mat_batch(coords, norm=norm)
    assert norm_dist_mat.shape == dist_mat.shape

    dist_mat_rel = torch.where(
        norm_dist_mat != 0,
        torch.divide(dist_mat, norm_dist_mat),
        fill
    )
    assert dist_mat_rel.shape == dist_mat.shape

    # instead of normalize the mean to 1, we normalize the mean of log(x) to 0
    eps = 1e-10
    log_mean = torch.mean(torch.log(dist_mat_rel + eps), dim=(1, 2))
    assert log_mean.shape == (I,)
    log_mean_rep = log_mean[:, None, None]
    dist_mat_rel /= torch.exp(log_mean_rep)
    
    # always set diagonal to 1
    dist_mat_rel[:,range(N),range(N)] = 1

    assert dist_mat_rel.shape == (I, N, N)
    return dist_mat_rel


def get_normalize_dist_mat(dist_mat):
    """
    Normalize the distance matrix by dividing the max distance.
    """
    assert dist_mat.shape[0] == dist_mat.shape[1]
    l, u = dist_mat.min(), dist_mat.max()
    eps = 1e-3
    return (dist_mat - l) / max(u - l, eps)


def get_normalize_dist_mat_batch(dist_mat):
    raise NotImplementedError


def get_normalize_coords(coords):
    """
    rescale the coords into the unit square [0,1] x [0,1]
    """

    assert coords.shape[1] == 2
    coords_min = coords.min(axis=0)
    coords_max = coords.max(axis=0)
    scale = coords_max.max() - coords_min.min()
    # scale is a scalar: must apply the same factor to both x & y, so that the optimal tour is not changed
    coords_normalized = (coords - coords_min) / scale

    # by doing this, we make sure that coords are in [0,1] x [0,1]
    # however, if the original data have larger range on x or y, 
    # then the normalized data are within a very narrow range, e.g., [0, 0.1] x [0, 1]
    # FIXME: should we do another way, e.g., scale to roughly [0,0.3] x [0,3]
    # FIXME: another case is usually, the [station] is an outlier, large spaces are empty

    return coords_normalized


def get_normalize_coords_batch(coords):
    raise NotImplementedError


def get_random_graph(n: int, num_graphs: int, non_Euc=True, rescale=False, seed=None):
    """
    Creates a batch of num_graphs random graph with N vertices.
    In is always in a "standard" distribution, but we also generate "scale_factors" to transform it to a more realistic/diverse instance

    args:
        num_graphs: batch size
        n: number of vertices in graph
        non_Euc: if True, return coords, rel_dist_mat, dist_mat, rescale_factors; else, return coords, rescale_factors
        rescale: if True, sample scale_factors from a log-normal distribution, indicates its deviation from the standard distribution
        seed: random seed (set as None in training, since it should already be set OUTSIDE)
    
    return: a dict with keys
        coords: shape = (num_graphs, n, 2)
        rel_distance: shape = (num_graphs, n, n), if non_Euc=True
        distance: shape = (num_graphs, n, n), if non_Euc=True
        scale_factors: shape = (num_graphs, 3) if non_Euc=True, else (num_graphs,1)

    """

    # in training, we should NOT use the same seed for all instances / batchs / epoches ...
    # for reproducibility, set the seed OUTSIDE (for the entire training process)
    if seed is not None:
        torch.manual_seed(seed)

    # for a real instance, we add three paramters to control its distribution:
    # (1) sacle along y-axis (stretch coords vertically)
    # (2) actaul_sym_std / sym_std, and (3) actual_asym_std / asym_std
    # that is: we can recover the original data after transformed to our standard distribution with above three parameters

    rescale_factors_dim = 3 if non_Euc else 1
    if rescale:
        # sd = 0.5 -> 50% between exp(-0.5) and exp(0.5), i.e., (0.61, 1.65)
        scale_factors = LogNormal(torch.tensor(0.0), torch.tensor(0.5)).sample(
            sample_shape=(num_graphs, rescale_factors_dim))
        assert scale_factors.shape == (num_graphs, rescale_factors_dim)
    else:
        scale_factors = None


    # Generate points randomly in unit square according to
    points = Uniform(0, 1).sample(sample_shape=(num_graphs, n, 2))
    assert points.shape == (num_graphs, n, 2)

    euclidean_distance_matrix = get_euclidean_dist_matrix(points)
    assert euclidean_distance_matrix.shape == (num_graphs, n, n)

    if not non_Euc:
        return {"coords": points, "scale_factors": scale_factors}

    # Yi: we decompose the relative matrix into two parts:
    # (1) a symmetric matrix, X1, and (2) a matrix represents the "asymmetry", X2,
    # so, relative matrix X = X1 * (1+X2)
    # further, element in X1 follows a [log-normal] distribution, mu=0, sigma=sym_std(=0.5)
    #       and element in X2 follows a [normal] distribution, mu=0, sigma=asym_std(=0.05)

    sym_std = 0.5
    sym_log = Normal(loc=0, scale=1).sample(sample_shape=(num_graphs, n, n))
    sym_rel_dist_mat = torch.exp(sym_std * np.sqrt(0.5) * (sym_log + sym_log.permute(0, 2, 1)))

    asym_std = 0.05
    asym = Normal(loc=0, scale=1).sample(sample_shape=(num_graphs, n, n))
    asym_rel_dist_mat = asym_std * np.sqrt(0.5) * (asym - asym.permute(0, 2, 1))

    relative_dist_matrix = sym_rel_dist_mat * (1 + asym_rel_dist_mat)
    assert relative_dist_matrix.shape == (num_graphs, n, n)

    distance_matrix = relative_dist_matrix * euclidean_distance_matrix

    return {"coords": points, "rel_distance": relative_dist_matrix, "distance": distance_matrix,
            "scale_factors": scale_factors}


def get_random_graph_np(*args, **kwargs):
    return to_np(get_random_graph(*args, **kwargs))


def scale_graph(data, sym_std=0.5, asym_std=0.05):

    assert isinstance(data, dict)
    coords = data["coords"]
    (I, N, _) = coords.shape
    if data.get("scale_factors") is None:
        mode = "normalize"
    else:
        assert len(data["scale_factors"].shape) == 2
        assert data["scale_factors"].shape[0] == I
        mode = "recover"
    
    # scale coords
    if mode == "recover":   # after recovery, coords are already in [0,1] x [0,1]
        r = data["scale_factors"][:,0]    # shape = (I)
        # TODO: combine with the else branch
        assert r.shape == (I,)
        r_x = torch.minimum(torch.ones_like(r), 1/r)[:,None]  # shape = (I,1)
        r_y = torch.minimum(torch.ones_like(r), r)[:,None] # shape = (I,1)
        coords = coords * torch.cat([r_x, r_y], dim=1)[:,None,:]
    else:
        coords_min = torch.min(coords, dim=1)[0]    # shape = (I,2)
        coords_max = torch.max(coords, dim=1)[0]    # shape = (I,2)
        assert coords_min.shape == (I, 2)
        r = ((coords_max[:, 1] - coords_min[:, 1]) / (coords_max[:, 0] - coords_min[:, 0]))
        assert r.shape == (I,)
        r_x = torch.maximum(torch.ones_like(r), r)[:,None]  # shape = (I,1)
        r_y = torch.maximum(torch.ones_like(r), 1/r)[:,None] # shape = (I,1)
        coords = coords * torch.cat([r_x, r_y], dim=1)[:,None,:]

    if "distance" not in data:  # Euclidean case
        scale_factors = r[:,None] if mode == "normalize" else None
        return {
            "coords": coords,
            "scale_factors": scale_factors
        }

    # non-Euclidean case
    dist_mat = data["distance"]
    dist_mat_rel = data["rel_distance"]
    (I, N, _) = dist_mat.shape

    eps = 1e-10
    dist_mat_rel = torch.maximum(dist_mat_rel, torch.ones_like(dist_mat_rel) * eps)
    dist_mat_rel_t = torch.transpose(dist_mat_rel, 1, 2)
    rel_sym = (dist_mat_rel + dist_mat_rel_t) / 2
    rel_asym = (dist_mat_rel - rel_sym) / rel_sym

    if mode=="recover":
        # note: scale_factors is actually, e.g., actual_sym_std / sym_std, NOT actual_sym_std
        sym_factor = data["scale_factors"][:,1]
        asym_factor = data["scale_factors"][:,2]
        assert sym_factor.shape == (I,)
    else:
        actual_sym_std = torch.std(torch.log(rel_sym), dim=(1, 2))
        actual_asym_std = torch.std(rel_asym, dim=(1, 2))
        sym_factor = sym_std / actual_sym_std
        asym_factor = asym_std / actual_asym_std
        assert sym_factor.shape == (I,)

    rel_sym = torch.exp(sym_factor[:, None, None] * torch.log(rel_sym))
    rel_asym = asym_factor[:, None, None] * rel_asym

    assert rel_sym.shape == rel_asym.shape == dist_mat_rel.shape

    dist_mat_rel = rel_sym * (1 + rel_asym)
    dist_mat = dist_mat_rel * get_euclidean_dist_matrix(coords)
    
    if mode == "recover":
        scale_factors = None
    else:
        scale_factors = torch.vstack([r, actual_sym_std/sym_std, actual_asym_std/asym_std]).T
        assert scale_factors.shape == (I, 3)
    
    return {
        "coords": coords,
        "distance": dist_mat,
        "rel_distance": dist_mat_rel,
        "scale_factors": scale_factors
    }


def normalize_graph(data, rescale=False):
    """
    args:
        data: unified input dict
            coords: (I, N, 2) torch tensor
            distance: (I, N, N) tensor
            rel_distance: (I, N, N) tensor
            scale_factors: (I, 3) if non_Euc else None?
    """

    if data.get("scale_factors") is not None:
        data = recover_graph(data)

    normalized_data = {}

    coords = data["coords"]
    assert len(coords.shape) == 3
    (I, N, _) = coords.shape

    # scale distance matrix to normalized coords
    coords_min = torch.min(coords, dim=1)[0]
    coords_max = torch.max(coords, dim=1)[0]
    assert coords_min.shape == (I, 2)

    scale = torch.max(coords_max, dim=1)[0] - torch.min(coords_min, dim=1)[0]
    assert scale.shape == (I,)

    coords_min_rep = coords_min[:, None, :]
    scale_rep = scale[:, None, None]
    coords = (coords - coords_min_rep) / scale_rep
    assert coords.shape == (I, N, 2)
    
    normalized_data["coords"] = coords
    
    if "distance" in data:
        dist_mat = data["distance"]
        dist_mat = dist_mat / scale_rep
        dist_mat_rel = get_rel_dist_mat_batch(coords, dist_mat)

        # FIXME: I believe the commented code below is actually taken care of in get_rel_dist_mat_batch
        # rel_norm_scale = torch.exp(torch.sum(torch.log(dist_mat_rel), dim=0) / N**2)
        # dist_mat_rel = dist_mat_rel / rel_norm_scale

        assert dist_mat_rel.shape == (I, N, N)
        dist_mat = dist_mat_rel * get_euclidean_dist_matrix(coords)
        normalized_data["distance"] = dist_mat
        normalized_data["rel_distance"] = dist_mat_rel
    normalized_data["scale_factors"] = None

    if rescale:
        return scale_graph(normalized_data)
    return normalized_data


def recover_graph(data):
    """
    Recovers the true distance matrix of a standardized graph (batched)
    """
    assert isinstance(data, dict)
    if data.get("scale_factors") is None:
        return data
    return scale_graph(data)


def normalize_graph_np(*args, **kwargs):
    return to_np(normalize_graph(*args, **kwargs))


def recover_graph_np(*args, **kwargs):
    return to_np(recover_graph(*args, **kwargs))


@deprecated
def tour_len(*args, **kwargs):
    return get_tour_len(*args, **kwargs)


@deprecated
def normed_dist_mat(*args, **kwargs):
    return get_normed_dist_mat(*args, **kwargs)


@deprecated
def normalize_dist_mat(*args, **kwargs):
    return get_normalize_dist_mat(*args, **kwargs)


@deprecated
def rel_dist_mat(*args, **kwargs):
    return get_rel_dist_mat(*args, **kwargs)


@deprecated
def normalize_coords(*args, **kwargs):
    return get_normalize_coords(*args, **kwargs)


@deprecated
def random_graph(*args, **kwargs):
    return get_random_graph(*args, **kwargs)
