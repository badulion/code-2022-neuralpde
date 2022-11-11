import numpy as np
import xarray as xr
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import torch
import os
from fnmatch import fnmatch


class PlasimDataset(Dataset):
    def __init__(self,
                 path,
                 target_dict,
                 years,
                 history=1,
                 horizon=4,
                 norm_years=[0]):

        # hardcoded for now
        self.norm_years = norm_years

        # parameters
        self.target_list = list(target_dict.keys())
        self.target_dict = target_dict
        self.years = years
        self.norm_years = norm_years
        self.history = history
        self.horizon = horizon
        self.path = path

        # lazy loading the data
        self.data = self._prepare_data(self.years)

        # calculating the normalizing parameters
        self.normalizer = {target: self._prepare_normalizer(target, self.norm_years) for target in self.target_list}

    def _prepare_data(self, years):
        file_list = self._list_paths(self.path, years)
        data = xr.open_mfdataset(file_list, combine='nested', concat_dim='time')
        data= data.isel(lev=0)
        return data

    def _prepare_normalizer(self, target, norm_years):
        norm_data = self._prepare_data(norm_years)
        var = self.target_dict[target] # variable 
        normalizer = [norm_data[var].mean().values.item(), norm_data[var].std().values.item()]
        return normalizer

    def _list_paths(self, path, years):
        file_list = os.listdir(path)
        norm_years_list = []
        for year in years:
            for file in file_list:
                pattern = f'*{year}_plevel.nc'
                if fnmatch(file, pattern):
                    norm_years_list.append(os.path.join(path, file))
                    break
        return norm_years_list

    def __getitem__(self, i: int) -> tuple:
        # features
        # x = self.calculate_diffs(arr)
        x = [self._get_var_x(target, i) for target in self.target_list]
        x = np.stack(x)
        x = np.concatenate(x, axis=1)

        # targets
        # y = np.expand_dims(arr, axis=1)
        y = [self._get_var_y(target, i) for target in self.target_list]
        y = np.concatenate(y, axis=1)

        return x, y, np.arange(self.horizon)+1


    def _get_var_x(self, target, i: int) -> tuple:
        # attributes
        normalizer = self.normalizer[target]
        variable = self.target_dict[target]

        # data
        arr = self.data.isel(time=slice(i, i+self.history))[variable].values
        arr = (arr-normalizer[0])/normalizer[1]
        arr = np.expand_dims(arr, axis=1)
        return arr

    def _get_var_y(self, target, i: int) -> tuple:
        # attributes
        normalizer = self.normalizer[target]
        variable = self.target_dict[target]

        # data
        arr = self.data.isel(time=slice(i+self.history, i+self.history + self.horizon))[variable].values
        arr = (arr-normalizer[0])/normalizer[1]
        arr = np.expand_dims(arr, axis=1)
        return arr

    def _calculate_diffs(self, ts):
        diffs = [ts[-1]]

        for i in range(self.max_diff):
            ts = ts[1:] - ts[:-1]
            diffs.append(ts[-1])

        diffs = [diffs[i] for i in self.differences]
        return diffs  # np.stack(diffs, axis=0)

    def __len__(self) -> int:
        return self.data.sizes["time"]-(self.history-1+self.horizon)


class PlasimDatamodule(pl.LightningDataModule):
    def __init__(self,
                 data_dir,
                 target_dict,
                 train_val_test_split = [0, 100, 105, 110],
                 batch_size: int = 8,
                 histories=[1,1,1],
                 num_workers=0,
                 horizons=[4,4,4],
                 norm_years=[0],
                 pin_memory: bool = False,
                 name="Weatherbench"):
        super().__init__()

        # variables and data
        self.path = data_dir
        self.target_dict = target_dict
        self.norm_years = norm_years


        self.history_train = histories[0]
        self.history_val = histories[1]
        self.history_test = histories[2]

        self.horizon_train = horizons[0]
        self.horizon_val = horizons[1]
        self.horizon_test = horizons[2]

        # loader specific:
        self.batch_size = batch_size
        self.num_workers = num_workers

        # train test val split
        self.train_years = range(train_val_test_split[0], train_val_test_split[1])
        self.val_years = range(train_val_test_split[1], train_val_test_split[2])
        self.test_years = range(train_val_test_split[2], train_val_test_split[3])


    def setup(self, stage=None):
        self.train = PlasimDataset(self.path,
                                   self.target_dict,
                                   self.train_years,
                                   history=self.history_train,
                                   horizon=self.horizon_train,
                                   norm_years=self.norm_years)
        self.val = PlasimDataset(self.path,
                                 self.target_dict,
                                 self.val_years,
                                 history=self.history_val,
                                 horizon=self.horizon_val,
                                 norm_years=self.norm_years)
        self.test = PlasimDataset(self.path,
                                  self.target_dict,
                                  self.test_years,
                                  history=self.history_test,
                                  horizon=self.horizon_test,
                                  norm_years=self.norm_years)

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