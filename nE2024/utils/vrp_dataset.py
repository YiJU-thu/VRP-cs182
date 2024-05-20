import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

import os, sys
import pickle
curr_path = os.path.dirname(__file__)
utils_vrp_path = os.path.join(curr_path, '..', '..', 'utils_project')
if utils_vrp_path not in sys.path:
    sys.path.append(utils_vrp_path)
from utils_vrp import get_random_graph, normalize_graph, recover_graph,\
      get_tour_len_torch, to_torch



class VRPTransformations:

    def roll(sol, shift=0):
        sol = sol.roll(shift, dims=-1)   # shift the solution, should remain optimal
        return sol


    def flip(coords, flip_type=0):
        # flip_type:
        # 0: (x, y), 1: (y, x), 2: (x, 1-y), 3: (y, 1-x),
        # 4: (1-x, y), 5: (1-y, x), 6: (1-x, 1-y), 7: (1-y, 1-x)
        
        squeeze_later = False
        if len(coords.shape) == 2:
            coords = coords.unsqueeze(0)    # add batch dimension
            squeeze_later = True

        x, y = coords[:, :, 0], coords[:, :, 1]

        if flip_type == 0:
            coords = (x, y)
        elif flip_type == 1:
            coords = (y, x)
        elif flip_type == 2:
            coords = (x, 1 - y)
        elif flip_type == 3:
            coords = (y, 1 - x)
        elif flip_type == 4:
            coords = (1 - x, y)
        elif flip_type == 5:
            coords = (1 - y, x)
        elif flip_type == 6:
            coords = (1 - x, 1 - y)
        elif flip_type == 7:
            coords = (1 - y, 1 - x)
        else:
            raise ValueError("Invalid flip type")
        coords = torch.stack(coords, dim=-1)

        if squeeze_later:
            coords = coords.squeeze(0)
        return coords
    

    def rotate(coords, angle=0):
        # angle: in radians
        
        squeeze_later = False
        if len(coords.shape) == 2:
            coords = coords.unsqueeze(0)    # add batch dimension
            squeeze_later = True
        
        rot_mat = torch.tensor([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]], dtype=coords.dtype)
        coords = torch.matmul(coords-0.5, rot_mat) + 0.5

        if squeeze_later:
            coords = coords.squeeze(0)
        return coords

    def transform(data, sol=None, flip_type=None, angle=None, shift=None):
        
        if flip_type is not None or angle is not None:
            coords = data.get('coords')
            if coords is None:
                # return data, sol
                raise ValueError("Coordinates not found in data. Cannot perform flip / rotate transformation.")
            
            squeeze_later = False
            if len(coords.shape) == 2:
                coords = coords.unsqueeze(0)    # add batch dimension
                squeeze_later = True
            
            if flip_type is not None:
                coords = VRPTransformations.flip(coords, flip_type)
            if angle is not None:
                coords = VRPTransformations.rotate(coords, angle)
            
            if squeeze_later:
                coords = coords.squeeze(0)
            data['coords'] = coords
        
        if shift is not None:
            sol = VRPTransformations.roll(sol, shift)
        
        return data, sol


class VRPDataset(Dataset):
    def __init__(self, filename=None, dataset=None, augmentation=None, n_aug=None, **kwargs):
        """
        augmentation: None, 'flip', 'rotate', 'flip_rotate'
        n_aug: number of augmentations to perform. If None, all possible augmentations are performed.
            else, sample n_aug augmentations randomly.
        """
        
        super(VRPDataset, self).__init__()

        assert not (filename is None and dataset is None), "Either filename or dataset must be provided"

        if filename is not None:
            data, solution = self._load_data_from_file(filename)
        elif dataset is not None:
            if isinstance(dataset, tuple):
                data, solution = dataset
            else:
                data = dataset
                solution = None
        else:
            # generate random data
            raise NotImplementedError("Random data generation not implemented")
        
        self.data = data
        self.solution = solution
        
        T = VRPTransformations
        if augmentation is None:
            self.n_transforms = 1
            self.transform = [lambda d, s: (d, s)]
        else:
            transforms = augmentation.split('_')
            assert all(t in ['flip', 'rotate', 'roll'] for t in transforms), "Invalid augmentation type"
            flip_types = list(range(8)) if 'flip' in transforms else [None]
            angles = [i*np.pi/4 for i in range(8)] if 'rotate' in transforms else [None]
            shifts = list(range(4)) if 'roll' in transforms else [None]
            self.n_transforms = len(flip_types) * len(angles) * len(shifts)
            self.transform = [lambda d, s: T.transform(d, s, flip_type=f, angle=a, shift=sh) 
                              for f in flip_types for a in angles for sh in shifts]

        if n_aug is not None:
            assert n_aug <= self.n_transforms
            self.transform = np.random.choice(self.transform, n_aug, replace=False)
            self.n_transforms = n_aug

        keep_rel = False
        if not keep_rel and "rel_distance" in self.data:
            del self.data["rel_distance"]

        if "coords" in self.data:
            self.size = self.data["coords"].shape[0]
        elif "distance" in self.data:
            self.size = self.data["distance"].shape[0]
        else:
            raise NotImplementedError("data has no 'coords' or 'distance' key")

    def __len__(self):
        return self.size * self.n_transforms
    
    def __getitem__(self, idx):
        idx_data = idx // self.n_transforms
        idx_transform = idx % self.n_transforms

        data = {}
        for k, v in self.data.items():
            data[k] = v[idx_data] if v is not None else torch.tensor([float('nan')])

        sol = None if self.solution is None else self.solution[idx_data]
        return self.transform[idx_transform](data, sol)
    

    def _load_data_from_file(self, filename):
        sol = None
        if isinstance(filename, dict):  # key: data fn, value: solution fn
            data_list = []
            sol_list = []
            for data_fn, sol_fn in filename.items():
                data_list.append(self._load_data_file(data_fn))
                sol_list.append(self._load_solution_file(sol_fn))
            data = self._merge_data(data_list)
            sol = self._merge_data(sol_list)
        elif isinstance(filename, list):    # only data files
            data_list = [self._load_data_file(fn) for fn in filename]
            data = self._merge_data(data_list)
        else:
            data = self._load_data_file(filename)
        return data, sol

    def _load_data_file(self, filename):
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        return to_torch(data)

    def _load_solution_file(self, filename):
        with open(filename, 'rb') as f:
            sol = pickle.load(f)["tour"]
        if isinstance(sol, list):
            sol = np.array(sol)   # convert to numpy array (training instances always have the same length ?)
        return torch.from_numpy(sol)

    def _merge_data(self, data_list):
        if isinstance(data_list[0], torch.Tensor):
            return torch.cat(data_list, dim=0)
        
        keys = data_list[0].keys()
        data = {}
        for key in keys:
            if isinstance(data_list[0][key], torch.Tensor):
                data[key] = torch.cat([d[key] for d in data_list], dim=0)
            # elif isinstance(data_list[0][key], np.ndarray):
            #     data[key] = np.concatenate([d[key] for d in data_list], axis=0)
            elif data_list[0][key] is None:
                data[key] = None
            else:
                raise TypeError("Data type not supported")
        return data
    

class VRPLargeDataset():
    def __init__(self, filenames, batch_size=500, n_loaded_files=10, start_file_idx=0, shuffle=True, augmentation=None, n_aug=None):
        
        if isinstance(filenames, list):
            self.filename_list = filenames
        elif isinstance(filenames, dict):
            self.filename_list = list(filenames.keys())
        else:
            raise ValueError("Invalid filenames")
        self.filenames = filenames
        
        self.n_total_files = len(self.filename_list)
        self.n_loaded_files = min(n_loaded_files, len(self.filename_list))
        self.pos = start_file_idx
        assert self.pos < self.n_total_files, "start_file_idx out of bounds"

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augmentation = augmentation
        self.n_aug = n_aug

        self._init_dataloader()

    def __iter__(self):
        while True:
            batch = next(self.dataloader, None)
            if batch is None:
                self._init_dataloader()
                batch = next(self.dataloader, None)
            yield batch

    def _init_dataloader(self):
        print("Loading files from", self.pos, "to", self.pos+self.n_loaded_files-1)
        
        filename_list = self.filename_list + self.filename_list # so index never goes out of bounds
        filename_toload = filename_list[self.pos:self.pos+self.n_loaded_files]
        
        if isinstance(self.filenames, list):
            filename = filename_toload
        elif isinstance(self.filenames, dict):
            filename = {fn: self.filenames[fn] for fn in filename_toload}
        else:
            raise ValueError("Invalid filenames")        
        
        dataset = VRPDataset(filename=filename, augmentation=self.augmentation, n_aug=self.n_aug)
        self.dataloader = iter(DataLoader(dataset, batch_size=self.batch_size, shuffle=self.shuffle))
        self.pos = (self.pos + self.n_loaded_files) % self.n_total_files