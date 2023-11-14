import numpy as np
from loguru import logger


def tour_len(tour, coords=None, dist_mat=None, norm=None):
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

def tour_len_batch(tour, coords=None, dist_mat=None, norm=None):
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


def normed_dist_mat(coords, norm="L2"):
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

def normed_dist_mat_batch(coords, norm="L2"):
    raise NotImplementedError


def rel_dist_mat(coords, dist_mat, norm="L2"):
    """
    Normalize the distance matrix relative to a simple normed distance, 
    i.e., how much the distance matrix deviates from 
        the one when the distance is a norm (e.g., "L2"), which info has effectively embedded in coords.
    
    args:
        coords (np.ndarray): shape = (N,2)
        dist_mat (np.ndarray): shape = (N,N)
        norm (str): "L1", "L2", "Linf", default = "L2"
    """

    assert coords.shape == (len(dist_mat), 2)
    assert dist_mat.shape == (len(dist_mat), len(dist_mat))
    assert norm in ["L1", "L2", "Linf"], "Invalid norm"

    expand = coords[:, None, :] - coords[None, :, :]

    if norm == "L2":
        norm_dist_mat = np.sqrt((expand ** 2).sum(axis=2))
    if norm == "L1":
        norm_dist_mat = np.abs(expand).sum(axis=2)
    if norm == "Linf":
        norm_dist_mat = np.max(np.abs(expand), axis=2)
    
    # find an OLS fit of dist_mat = k * norm_dist_mat
    # then normalize dist_mat = dist_mat / dist_mat_hat

    y = dist_mat.flatten()
    x = norm_dist_mat.flatten()
    k = np.dot(x, y) / np.dot(x, x)

    dist_mat_hat = k * norm_dist_mat
    dist_mat_rel = np.divide(dist_mat, dist_mat_hat, out=np.zeros_like(dist_mat), where=dist_mat_hat!=0)

    return dist_mat_rel


def normalize_dist_mat(dist_mat):
    """
    Normalize the distance matrix by dividing the max distance.
    """
    assert dist_mat.shape[0] == dist_mat.shape[1]
    l, u = dist_mat.min(), dist_mat.max()
    eps = 1e-3
    return (dist_mat-l) / max(u-l, eps)

def normalize_dist_mat_batch(dist_mat):
    raise NotImplementedError


def normalize_coords(coords):
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

def normalize_coords_batch(coords):
    raise NotImplementedError


def random_graph(N, I, dist_mat_params, seed=None):
    """
    Generate random coordinates in the unit square [0,1] x [0,1]
    """
    # in training, we should NOT use the same seed for all instances / batchs / epoches ...
    # for reproducibility, set the seed OUTSIDE (for the entire training process)
    if seed is not None:    
        np.random.seed(seed)
    coords = np.random.uniform(size=(I, N, 2))
    rel_dist_mat = ...  # TODO: what is the reasonable distribution to draw from?
    raise NotImplementedError

    dist_mat = rel_dist_mat * normed_dist_mat(coords, norm="L2")
    return coords, rel_dist_mat, dist_mat
    