import numpy as np
import xarray as xr
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import torch
import os
from fnmatch import fnmatch


class WeatherbenchDataset(Dataset):
    def __init__(self,
                 path,
                 target_dict,
                 years,
                 history=1,
                 horizon=4,
                 norm_years=[1979]):

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
        self.data = {target: self._prepare_dataset(target, self.years) for target in self.target_list}

        # calculating the normalizing parameters
        self.normalizer = {target: self._prepare_normalizer(target, self.norm_years) for target in self.target_list}

    def _prepare_dataset(self, target, years):
        file_list = self._list_paths(self.path, target, years)
        data = xr.open_mfdataset(file_list)
        return data

    def _prepare_normalizer(self, target, norm_years):
        norm_years_list = self._list_paths(self.path, target, norm_years)
        data = xr.open_mfdataset(norm_years_list)
        normalizer = [data[self.target_dict[target]].mean().values.item(), data[self.target_dict[target]].std().values.item()]
        return normalizer

    def _list_paths(self, path, target, years):
        target_path = os.path.join(path, target)
        file_list = os.listdir(target_path)
        norm_years_list = []
        for year in years:
            for file in file_list:
                pattern = f'{target}*{year}*'
                if fnmatch(file, pattern):
                    norm_years_list.append(os.path.join(target_path, file))
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
        data = self.data[target]
        normalizer = self.normalizer[target]
        variable = self.target_dict[target]

        # data
        arr = data.isel(time=slice(i, i+self.history))[variable].values
        arr = (arr-normalizer[0])/normalizer[1]
        arr = np.expand_dims(arr, axis=1)
        return arr

    def _get_var_y(self, target, i: int) -> tuple:
        # attributes
        data = self.data[target]
        normalizer = self.normalizer[target]
        variable = self.target_dict[target]

        # data
        arr = data.isel(time=slice(i+self.history, i+self.history + self.horizon))[variable].values
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
        target_sizes = [self.data[target].sizes["time"]-(self.history-1+self.horizon) for target in self.target_list]
        return min(target_sizes)


class WeatherbenchDatamodule(pl.LightningDataModule):
    def __init__(self,
                 data_dir,
                 target_dict,
                 train_val_test_split = [1979, 2015, 2017, 2019],
                 batch_size: int = 8,
                 histories=[1,1,1],
                 num_workers=0,
                 horizons=[4,4,4],
                 norm_years=[1979],
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
        self.train = WeatherbenchDataset(self.path,
                                         self.target_dict,
                                         self.train_years,
                                         history=self.history_train,
                                         horizon=self.horizon_train,
                                         norm_years=self.norm_years)
        self.val = WeatherbenchDataset(self.path,
                                       self.target_dict,
                                       self.val_years,
                                         history=self.history_val,
                                       horizon=self.horizon_val,
                                       norm_years=self.norm_years)
        self.test = WeatherbenchDataset(self.path,
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