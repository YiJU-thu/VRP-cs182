import torch
from torch.distributions.distribution import Distribution as Distribution
from torch.distributions.uniform import Uniform


def create_dataset(n: int, config: dict):
    """
    Creates a random graph with N vertices, and returns it
    as a dict (TODO).

    args:
        n: number of vertices in graph
        config: dataset configuration parameters, as dict
            - "mode": str
            - "point_distribution": a torch distribution
            - "deform_distribution": a torch distribution

    """

    if config["mode"] == "deform":
        # Generate points randomly in unit square according to
        # config["point_distribution"], and deform the distance matrix
        # according to config["deform_distribution"]

        x_distribution: Distribution = config["x_distribution"]
        y_distribution: Distribution = config["y_distribution"]
        x_coords = x_distribution.sample(sample_shape=(n, 1))
        y_coords = y_distribution.sample(sample_shape=(n, 1))
        points = torch.cat((x_coords, y_coords), dim=1)
        assert points.shape == (n, 2)

        euclidean_distance_matrix = torch.cdist(points, points, p=2)
        deform_dist: Distribution = config["deform_distribution"]
        deformation_matrix = deform_dist.sample(sample_shape=(n, n))

        distance_matrix = deformation_matrix * euclidean_distance_matrix
        return {
            "points": points,
            "distance_matrix": distance_matrix
        }
    else:
        raise NotImplementedError


# demo
if __name__ == '__main__':
    config = {
        "mode": "deform",
        "x_distribution": Uniform(torch.tensor(-0.5), torch.tensor(0.5)),
        "y_distribution": Uniform(torch.tensor(-0.5), torch.tensor(0.5)),
        "deform_distribution": Uniform(torch.tensor(0.5), torch.tensor(2.0))
    }
    dataset = create_dataset(20, config)
