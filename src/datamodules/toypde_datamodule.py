from encodings import normalize_encoding
from random import sample
from tracemalloc import start
import numpy as np
import xarray as xr
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import torch
import os


class PDEDataset(Dataset):
    def __init__(self,
                 path,
                 target_dict,
                 history=1,
                 horizon=4,
                 normalizer=None):

        # file paths
        self.path = path


        # parameters
        self.target_dict = target_dict
        self.target_list = list(target_dict.keys())
        self.history = history
        self.horizon = horizon

        # load the data
        self.data = self._prepare_data()
        self.num_observations, self.num_samples = self._count_observations()

        self.end_indices = np.cumsum(self.num_samples)
        self.start_indices = self.end_indices - self.num_samples


        # calculating the normalizing parameters
        if normalizer:
            self.normalizer = normalizer
        else:
            self.normalizer = {target: self._prepare_normalizer(target) for target in self.target_list}

    def _prepare_data(self):
        files = os.listdir(self.path)
        files = [os.path.join(self.path, file) for file in files]
        data = [xr.open_dataset(file) for file in files]
        return data

    def _count_observations(self, ):
        num_observations = [len(ds.time) for ds in self.data]
        num_samples = [n - (self.history-1+self.horizon) for n in num_observations]
        return np.array(num_observations), np.array(num_samples)

    def _prepare_normalizer(self, target):
        var = self.target_dict[target] # variable 
        mean = [ds[var].mean().values.item() for ds in self.data]
        std = [ds[var].std().values.item() for ds in self.data]

        normalizer = [
            np.sum(mean*self.num_observations)/np.sum(self.num_observations), 
            np.sum(std*self.num_observations)/np.sum(self.num_observations)
        ]
        return normalizer

    def __getitem__(self, i: int) -> tuple:
        ds_ix, rel_ix = self._get_dataset_index(i)


        # features
        # x = self.calculate_diffs(arr)
        x = [self._get_var_x(self.data[ds_ix], target, rel_ix) for target in self.target_list]
        x = np.stack(x)
        x = np.concatenate(x, axis=1)

        # targets
        # y = np.expand_dims(arr, axis=1)
        y = [self._get_var_y(self.data[ds_ix], target, rel_ix) for target in self.target_list]
        y = np.concatenate(y, axis=1)

        return x, y, np.arange(self.horizon)+1
    
    def _get_dataset_index(self, i: int) -> int:
        ds_index = np.argwhere((self.start_indices <= i)*(self.end_indices > i)).item()
        relative_index = i - self.start_indices[ds_index]
        return ds_index, relative_index
        

    def _get_var_x(self, ds, target, i: int) -> tuple:
        # attributes
        #data = self.data[target]
        normalizer = self.normalizer[target]
        variable = self.target_dict[target]

        # data
        arr = ds.isel(time=slice(i, i+self.history))[variable].values
        arr = (arr-normalizer[0])/normalizer[1]
        arr = np.expand_dims(arr, axis=1)
        return arr

    def _get_var_y(self, ds, target, i: int) -> tuple:
        # attributes
        #data = self.data[target]
        normalizer = self.normalizer[target]
        variable = self.target_dict[target]

        # data
        arr = ds.isel(time=slice(i+self.history, i+self.history + self.horizon))[variable].values
        arr = (arr-normalizer[0])/normalizer[1]
        arr = np.expand_dims(arr, axis=1)

        return arr

    def __len__(self) -> int:
        return np.sum(self.num_samples, dtype=np.int32)


class PDEDatamodule(pl.LightningDataModule):
    def __init__(self,
                 data_dir,
                 target_dict,
                 batch_size: int = 8,
                 histories=[1,1,1],
                 num_workers=0,
                 horizons=[4,4,4],
                 pin_memory: bool = False):
        super().__init__()

        # variables and data
        self.path = data_dir

        self.target_dict = target_dict

        self.history_train = histories[0]
        self.history_val = histories[1]
        self.history_test = histories[2]

        self.horizon_train = horizons[0]
        self.horizon_val = horizons[1]
        self.horizon_test = horizons[2]

        # loader specific:
        self.batch_size = batch_size
        self.num_workers = num_workers


    def setup(self, stage=None):
        self.train = PDEDataset(os.path.join(self.path, "train"),
                                self.target_dict,
                                history=self.history_train,
                                horizon=self.horizon_train)
        self.val = PDEDataset(os.path.join(self.path, "val"),
                                self.target_dict,
                                history=self.history_val,
                                horizon=self.horizon_val)
        self.test = PDEDataset(os.path.join(self.path, "test"),
                                self.target_dict,
                                history=self.history_test,
                                horizon=self.horizon_test)

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=self.collate_fn, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=self.collate_fn, shuffle=True)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=self.collate_fn)

    def collate_fn(self, sample_list):
        x = np.stack([item[0] for item in sample_list], axis=1)
        y = np.stack([item[1] for item in sample_list], axis=1)
        #t = np.stack([item[2] for item in sample_list], axis=1)
        t = sample_list[0][2]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32), torch.tensor(t, dtype=torch.float32)


if __name__ == '__main__':

    target_dict = {
        'density': 'p',
        'temperature': 't',
        'velocity_x': 'u',
        'velocity_y': 'v',
    }
    ds = PDEDataset(path = "data/toypdes/gas_dynamics/train", target_dict=target_dict)

