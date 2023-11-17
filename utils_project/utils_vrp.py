import numpy as np
from loguru import logger
import torch
from torch.distributions.distribution import Distribution as Distribution
from torch.distributions.uniform import Uniform

def deprecated(func):
    def new_func(*args, **kwargs):
        logger.warning(f"Call to deprecated function {func.__name__}. Use get_{func.__name__} instead.")
        return func(*args, **kwargs)
    return new_func



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
    raise NotImplementedError


def get_rel_dist_mat(coords, dist_mat, norm="L2", fill=1):
    """
    Normalize the distance matrix relative to a simple normed distance, 
    i.e., how much the distance matrix deviates from 
        the one when the distance is a norm (e.g., "L2"), which info has effectively embedded in coords.
    
    args:
        coords (np.ndarray): shape = (N,2)
        dist_mat (np.ndarray): shape = (N,N)
        norm (str): "L1", "L2", "Linf", default = "L2"
        fill (float): fill the relative distance matrix with this value where face x/0, default = 1
    """

    assert coords.shape == (len(dist_mat), 2)
    assert dist_mat.shape == (len(dist_mat), len(dist_mat))
    
    norm_dist_mat = get_normed_dist_mat(coords, norm=norm)
    
    # find an OLS fit of dist_mat = k * norm_dist_mat
    # then normalize dist_mat = dist_mat / dist_mat_hat

    # FIXME: we may skip the dist_mat_hat part, just divide by norm_dist_mat,
    # the only concern is: in this case, should we still fill the diagonal with 1 (before or after normalization)? 
    y = dist_mat.flatten()
    x = norm_dist_mat.flatten()
    k = np.dot(x, y) / np.dot(x, x)

    dist_mat_hat = k * norm_dist_mat
    dist_mat_rel = np.divide(dist_mat, dist_mat_hat, out=np.ones_like(dist_mat)*fill, where=dist_mat_hat!=0)
    dist_mat_rel /= dist_mat_rel.mean() # normalize the mean to 1
    return dist_mat_rel


def get_normalize_dist_mat(dist_mat):
    """
    Normalize the distance matrix by dividing the max distance.
    """
    assert dist_mat.shape[0] == dist_mat.shape[1]
    l, u = dist_mat.min(), dist_mat.max()
    eps = 1e-3
    return (dist_mat-l) / max(u-l, eps)

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


def random_graph(n: int, num_graphs: int, config: dict = None, seed=None):
    """
    Creates a random graph with N vertices, by default random coordinates in
    [0, 1] x [0, 1]

    args:
        num_graphs: batch size
        n: number of vertices in graph
        config: dataset configuration parameters, as dict
            - "mode": str
            - "seed": int or None
            - "point_distribution": a torch distribution
            - "deform_distribution": a torch distribution

    """

    if not config:
        # default config
        config = {
            "mode": "deform",
            "x_distribution": Uniform(torch.tensor(0), torch.tensor(1)),
            "y_distribution": Uniform(torch.tensor(0), torch.tensor(1)),
            "deform_distribution": Uniform(torch.tensor(0.5), torch.tensor(2.0))
        }

    # TODO: don't fully understand - should seed be set here?
    # following example of original random_graph funcition, but not sure
    # in training, we should NOT use the same seed for all instances / batchs / epoches ...
    # for reproducibility, set the seed OUTSIDE (for the entire training process)
    if seed is not None:
        torch.manual_seed(seed)

    if config["mode"] == "deform":
        # Generate points randomly in unit square according to
        # config["point_distribution"], and deform the distance matrix
        # according to config["deform_distribution"]

        x_distribution: Distribution = config["x_distribution"]
        y_distribution: Distribution = config["y_distribution"]
        x_coords = x_distribution.sample(sample_shape=(num_graphs, n, 1))
        y_coords = y_distribution.sample(sample_shape=(num_graphs, n, 1))
        points = torch.cat((x_coords, y_coords), dim=-1)
        assert points.shape == (num_graphs, n, 2)

        euclidean_distance_matrix = torch.cdist(points, points, p=2)
        assert euclidean_distance_matrix.shape == (num_graphs, n, n)

        deform_dist: Distribution = config["deform_distribution"]
        relative_dist_matrix = deform_dist.sample(sample_shape=(num_graphs, n, n))

        distance_matrix = relative_dist_matrix * euclidean_distance_matrix
        return points, relative_dist_matrix, distance_matrix

    else:
        raise NotImplementedError





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


