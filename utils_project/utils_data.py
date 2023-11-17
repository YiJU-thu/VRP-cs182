import torch
from torch.distributions.distribution import Distribution as Distribution
from torch.distributions.uniform import Uniform
import numpy as np
from loguru import logger

"""
THIS NOTEBOOK WILL BE REMOVED LATER (Yi)
"""

# def create_dataset(n: int, config: dict):
#     """
#     Creates a random graph with N vertices, and returns it
#     as a dict (TODO).
#
#     args:
#         n: number of vertices in graph
#         config: dataset configuration parameters, as dict
#             - "mode": str
#             - "point_distribution": a torch distribution
#             - "deform_distribution": a torch distribution
#
#     """
#
#     if config["mode"] == "deform":
#         # Generate points randomly in unit square according to
#         # config["point_distribution"], and deform the distance matrix
#         # according to config["deform_distribution"]
#
#         x_distribution: Distribution = config["x_distribution"]
#         y_distribution: Distribution = config["y_distribution"]
#         x_coords = x_distribution.sample(sample_shape=(n, 1))
#         y_coords = y_distribution.sample(sample_shape=(n, 1))
#         points = torch.cat((x_coords, y_coords), dim=1)
#         assert points.shape == (n, 2)
#
#         euclidean_distance_matrix = torch.cdist(points, points, p=2)
#         deform_dist: Distribution = config["deform_distribution"]
#         deformation_matrix = deform_dist.sample(sample_shape=(n, n))
#
#         distance_matrix = deformation_matrix * euclidean_distance_matrix
#         return {
#             "points": points,
#             "distance_matrix": distance_matrix
#         }
#     else:
#         raise NotImplementedError

def create_dataset(num_graphs: int, n: int, config: dict = None):
    """
    Creates a random graph with N vertices, and returns it
    as a dict (TODO).

    args:
        num_graphs: batch size
        n: number of vertices in graph
        config: dataset configuration parameters, as dict
            - "mode": str
            - "seed": int or None
            - "point_distribution": a torch distribution
            - "deform_distribution": a torch distribution

    """
    logger.error("this method is abondoned, use utils_vrp/get_random_graph instead")
    raise NotImplementedError
    if not config:
        # default config
        config = {
            "mode": "deform",
            "x_distribution": Uniform(torch.tensor(0.0), torch.tensor(1.0)),
            "y_distribution": Uniform(torch.tensor(0.0), torch.tensor(1.0)),
            "seed": None,
            "deform_distribution": Uniform(torch.tensor(0.5), torch.tensor(2.0))
        }

    # TODO: don't fully understand - should seed be set here?
    # if config["seed"] is not None:
    #     torch.manual_seed(config["seed"])

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


# demo
if __name__ == '__main__':
    dataset = create_dataset(num_graphs=4, n=20, config=None)
    points, rel_dist, dist = dataset

