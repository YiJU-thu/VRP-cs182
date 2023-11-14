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


def normalize_dist_mat(coords, dist_mat, norm="L2"):
    """
    Normalize the distance matrix, i.e., how much the distance matrix deviates from 
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
    dist_mat_normalized = np.divide(dist_mat, dist_mat_hat, out=np.zeros_like(dist_mat), where=dist_mat_hat!=0)

    return dist_mat_normalized


