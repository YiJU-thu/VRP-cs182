import time
import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.utils import shuffle
import pickle


def rotate(raw, center = np.array([0.5, 0.5])):
    
    raw = np.array(raw)
    new = np.zeros_like(raw)
    angle = np.random.randint(low = -180, high = 180)
    
    new[:,0] = (raw[:,0] - center[0]) * np.cos(angle) - (raw[:,1] - center[1]) * np.sin(angle) + center[0]
    new[:,1] = (raw[:,0] - center[0]) * np.sin(angle) + (raw[:,1] - center[1]) * np.cos(angle) + center[1]
    
    minmum = np.minimum(np.min(new), 0)
    new -= minmum
    maximum = np.maximum(np.max(new), 1)
    new /= maximum
    #print(angle, minmum, maximum)
    return new


def edge_target_batch(tours):
    """
    Given a batch of tours (permutation of nodes), return a batch of 0-1 edge targets.
    connect_mat[i,j] = 1 iff (i,j) or (j,i) is an edge in the tour.
    
    args:
        tours: a batch of tours, shape (B, V)
    returns:
        connect_mat: a batch of edge targets, shape (B, V, V)
    """

    B, V = tours.shape

    # Create an index array for the columns and rows
    indices = np.indices((B, V))

    # Set the corresponding elements in the adjacency matrices to 1
    connect_mat = np.zeros((B,V,V), dtype=int)
    # index (b, tours[b,i], tours[b,i+1]) - indexing in batch
    connect_mat[indices[0], tours, np.roll(tours, 1, axis=1)]=1
    connect_mat[indices[0], tours, np.roll(tours, -1, axis=1)]=1

    return connect_mat


class DotDict(dict):
    """Wrapper around in-built dict class to access members through the dot operation.
    """

    def __init__(self, **kwds):
        self.update(kwds)
        self.__dict__ = self


class GoogleTSPReader(object):
    """Iterator that reads TSP dataset files and yields mini-batches.
    
    Format expected as in Vinyals et al., 2015: https://arxiv.org/abs/1506.03134, http://goo.gl/NDcOIG
    """

    def __init__(self, num_nodes, num_neighbors, batch_size, filepath, shuffled = True, augmentation = False, aug_prob = 0.9):
        """
        Args:
            num_nodes: Number of nodes in TSP tours
            num_neighbors: Number of neighbors to consider for each node in graph
            batch_size: Batch size
            filepath: Path to dataset file (.txt file) or .pkl file (TODO)
        """
        self.num_nodes = num_nodes
        self.num_neighbors = num_neighbors
        self.batch_size = batch_size
        self.filepath = filepath
        self.augmentation = augmentation
        self.aug_prob = aug_prob

        if isinstance(filepath, tuple):
            filepath, sol_path = filepath   # instances & solutions are stored in two files
            # b/c instance can be generated quickly given the random seed, and is much larger in size (B X V X V)
            # while solution is much smaller in size (B X V), and takes much longer to generate, so we only track solutions in git


        if ".pkl" in filepath:
            self.file_type = "pkl"
            with open(filepath, "rb") as f:
                dataset = pickle.load(f)
                # this is a dict containing keys: "coords", "distance", "rel_distance", "scale_factors" (np.ndarray)
            with open(sol_path, "rb") as f:
                sol = pickle.load(f)
                # this is a dict containing keys: "obj", "time", "tour" (np.ndarray or list)
            tour_len = sol["obj"]
            tour = np.array(sol["tour"])
            assert tour.shape == dataset["coords"].shape[:2], f'tour.shape = {tour.shape}'

            if shuffled:
                if dataset.get("scale_factors") is not None:
                    coords, distance, rel_distance, scale_factors, tour, tour_len =\
                         shuffle(dataset["coords"], dataset["distance"], dataset["rel_distance"], dataset["scale_factors"], tour, tour_len)
                else:
                    coords, distance, rel_distance, tour, tour_len = shuffle(dataset["coords"], dataset["distance"], dataset["rel_distance"], tour, tour_len)
                    scale_factors = None
                dataset = {"coords": coords, "distance": distance, "rel_distance": rel_distance, "scale_factors": scale_factors,
                            "tour": tour, "tour_len": tour_len}
            else:
                dataset["tour"] = tour
                dataset["tour_len"] = tour_len
            
            self.filedata = dataset
            file_len = len(self.filedata["coords"])
        
        elif ".txt" in filepath:
            self.file_type = "txt"
            if shuffled:
                self.filedata = shuffle(open(filepath, "r").readlines())  # Always shuffle upon reading data
            else:
                self.filedata = open(filepath, "r").readlines()
            file_len = len(self.filedata)
        
        else:
            raise ValueError("Invalid file format. Must be .txt or .pkl file.")

        self.max_iter = (file_len // batch_size)

    def __iter__(self):
        for batch in range(self.max_iter):
            start_idx = batch * self.batch_size
            end_idx = (batch + 1) * self.batch_size
            yield self.process_batch(start_idx, end_idx)

    def process_batch(self, start_idx, end_idx):
        """Helper function to convert raw lines into a mini-batch as a DotDict.
        """
        
        if self.file_type == "pkl":
            batch = self._process_batch_pkl(start_idx, end_idx)
        if self.file_type == "txt":
            batch = self._process_batch_txt(start_idx, end_idx)
        
        B, V = self.batch_size, self.num_nodes
        assert batch.edges.shape == (B,V,V), f'batch.edges.shape = {batch.edges.shape}'
        assert batch.edges_values.shape == (B,V,V), f'batch.edges_values.shape = {batch.edges_values.shape}'
        assert batch.edges_target.shape == (B,V,V), f'batch.edges_target.shape = {batch.edges_target.shape}'
        assert batch.nodes.shape == (B,V), f'batch.nodes.shape = {batch.nodes.shape}'
        assert batch.nodes_coord.shape == (B,V,2), f'batch.nodes_coord.shape = {batch.nodes_coord.shape}'
        assert batch.tour_nodes.shape == (B,V), f'batch.tour_nodes.shape = {batch.tour_nodes.shape}'
        assert batch.tour_len.shape == (B,), f'batch.tour_len.shape = {batch.tour_len.shape}'

        return batch
        
    
    def _process_batch_pkl(self, start_idx, end_idx):

        if self.augmentation:
            raise ValueError("Augmentation has not been supported by .pkl files.")
            # NOTE: is augmentation necessary for .pkl files?

        # TODO
        # From list to tensors as a DotDict
        B, V = self.batch_size, self.num_nodes
        coords = self.filedata["coords"][start_idx:end_idx]
        distance = self.filedata["distance"][start_idx:end_idx]
        rel_distance = self.filedata["rel_distance"][start_idx:end_idx]
        scale_factors = self.filedata["scale_factors"][start_idx:end_idx] if self.filedata["scale_factors"] is not None else None

        tour = self.filedata["tour"][start_idx:end_idx]
        tour_len = self.filedata["tour_len"][start_idx:end_idx]
        
        if start_idx == 0:
            # TODO: check the datafile and the solution file really match with each other
            tour_len_recompute = np.sum(distance[np.indices((B,V))[0], tour, np.roll(tour, -1, axis=1)], axis=1)
            assert tour_len_recompute.shape == tour_len.shape, f'tour_len_recompute.shape = {tour_len_recompute.shape}'
            assert np.allclose(tour_len, tour_len_recompute), f'tour_len = {tour_len[:5]}, tour_len_recompute = {tour_len_recompute[:5]}'
            print("instance file and solution file match with each other")

        batch = DotDict()
        if self.num_neighbors == -1:
            batch.edges = np.ones((B,V,V)) # adjacent matrix: B x V x V
        else:
            raise NotImplementedError("num_neighbors != -1 has not been supported by .pkl files.")
        batch.edges_values = distance   # distance matrix: B x V x V
        batch.edges_target = edge_target_batch(tour) # connection matrix: B x V x V
        batch.nodes = np.ones((B,V)) # node features: B x V FIXME: how is this used?
        batch.nodes_target = np.argsort(tour, axis=1) # node targets: B x V. 
        # node_target[b, i] = j means the i-th node is the j-th to visit in the tour
        batch.nodes_coord = coords  # node coordinates: B x V x 2
        batch.tour_nodes =  tour # tour nodes: B x V
        batch.tour_len = tour_len # tour length: B

        batch.rel_edges_values = rel_distance
        batch.scale_factors = scale_factors

        return batch
    
    
    def _process_batch_txt(self, start_idx, end_idx):
        
        lines = self.filedata[start_idx:end_idx]
        
        batch_edges = []
        batch_edges_values = []
        batch_edges_target = []  # Binary classification targets (0/1)
        batch_nodes = []
        batch_nodes_target = []  # Multi-class classification targets (`num_nodes` classes)
        batch_nodes_coord = []
        batch_tour_nodes = []
        batch_tour_len = []

        for line_num, line in enumerate(lines):
            line = line.split(" ")  # Split into list
            
            # Compute signal on nodes
            nodes = np.ones(self.num_nodes)  # All 1s for TSP...
            
            # Convert node coordinates to required format
            nodes_coord = []
            for idx in range(0, 2 * self.num_nodes, 2):
                nodes_coord.append([float(line[idx]), float(line[idx + 1])])
            if self.augmentation:
                if np.random.uniform() > self.aug_prob:
                    #print(f'aug = {a}')
                    nodes_coord = rotate(raw = nodes_coord)
                #else:
                    #print(f'non-aug = {a}')
                
            # Compute distance matrix
            # FIXME: only accept Euclidean cases now
            # pdist returns a n*(n-1)/2 vector, so we need to use squareform to get the n^2 matrix
            W_val = squareform(pdist(nodes_coord, metric='euclidean'))
            
            # Compute adjacency matrix
            if self.num_neighbors == -1:
                W = np.ones((self.num_nodes, self.num_nodes))  # Graph is fully connected
            else:
                W = np.zeros((self.num_nodes, self.num_nodes))
                # Determine k-nearest neighbors for each node
                knns = np.argpartition(W_val, kth=self.num_neighbors, axis=-1)[:, self.num_neighbors::-1]
                # Make connections 
                for idx in range(self.num_nodes):
                    W[idx][knns[idx]] = 1
            np.fill_diagonal(W, 2)  # Special token for self-connections
            
            # Convert tour nodes to required format
            # Don't add final connection for tour/cycle
            tour_nodes = [int(node) - 1 for node in line[line.index('output') + 1:-1]][:-1]
            
            # Compute node and edge representation of tour + tour_len
            tour_len = 0
            nodes_target = np.zeros(self.num_nodes)
            edges_target = np.zeros((self.num_nodes, self.num_nodes))
            for idx in range(len(tour_nodes) - 1):
                i = tour_nodes[idx]
                j = tour_nodes[idx + 1]
                nodes_target[i] = idx  # node targets: ordering of nodes in tour
                edges_target[i][j] = 1
                edges_target[j][i] = 1
                tour_len += W_val[i][j]
            
            # Add final connection of tour in edge target
            nodes_target[j] = len(tour_nodes) - 1
            edges_target[j][tour_nodes[0]] = 1
            edges_target[tour_nodes[0]][j] = 1
            tour_len += W_val[j][tour_nodes[0]]
            
            # Concatenate the data
            batch_edges.append(W)
            batch_edges_values.append(W_val)
            batch_edges_target.append(edges_target)
            batch_nodes.append(nodes)
            batch_nodes_target.append(nodes_target)
            batch_nodes_coord.append(nodes_coord)
            batch_tour_nodes.append(tour_nodes)
            batch_tour_len.append(tour_len)
        
        # From list to tensors as a DotDict
        batch = DotDict()
        batch.edges = np.stack(batch_edges, axis=0)
        batch.edges_values = np.stack(batch_edges_values, axis=0)
        batch.edges_target = np.stack(batch_edges_target, axis=0)
        batch.nodes = np.stack(batch_nodes, axis=0)
        batch.nodes_target = np.stack(batch_nodes_target, axis=0)
        batch.nodes_coord = np.stack(batch_nodes_coord, axis=0)
        batch.tour_nodes = np.stack(batch_tour_nodes, axis=0)
        batch.tour_len = np.stack(batch_tour_len, axis=0)
        return batch
