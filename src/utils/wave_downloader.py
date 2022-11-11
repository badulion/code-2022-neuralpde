from hashlib import algorithms_guaranteed
import shutil
import numpy as np

import xarray as xr
import requests
import matplotlib.pyplot as plt
import pydap.client
import os
from rich.progress import Progress
import time

TIMES = range(0, 81808)
VARIABLES = ["VHM0", "VMDR", "VPED"]

class WaveDownloader:
    def __init__(self,
                 variables,
                 data_dir,
                 start_time,
                 end_time,
                 minibatch_size,
                 batch_size,
                 timeout=100,
                 continue_download=True) -> None:

        self.variables = variables
        self.data_dir = data_dir
        self.start_time = start_time
        self.end_time = end_time
        self.minibatch_size = minibatch_size
        self.batch_size = batch_size
        self.timeout = timeout
        self.continue_download = continue_download

        if not self.continue_download:
            shutil.rmtree(self.data_dir)

        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir, exist_ok=True)

    def _download_minibatch_variable(self, variable, start_time, end_time):
        url = f'https://my.cmems-du.eu/thredds/dodsC/cmems_mod_glo_wav_my_0.2_PT3H-i.dods?{variable}[{start_time}:1:{end_time}][0:1:898][0:1:1799]'
        session = requests.Session()
        session.auth = ('adulny', 'mam2DUZEstoly#')

        # actual download
        dataset = pydap.client.open_dods(url, session=session, timeout=self.timeout)[variable]

        # process
        data_batch = dataset.array[:].data
        data_batch = np.float32(data_batch)
        data_batch[data_batch< 0 ] = np.nan

        # convert to xarray
        dims = dataset.dimensions
        coords_dict = {
            'time': dataset.time.data, 
            'latitude': dataset.latitude.data, 
            'longitude': dataset.longitude.data
        }

        xr_dataarray = xr.DataArray(data_batch, dims=dims, coords=coords_dict)
        return xr_dataarray

    def _download_minibatch(self, variable_list, start_time, end_time):
        x_dataarrays = {
            var: self._download_minibatch_variable(var, start_time, end_time) for var in variable_list
        }

        x_dataset = xr.Dataset(
            x_dataarrays,
        )

        new_lat = np.arange(-87.1875, 90, step=5.625)
        new_lon = np.arange(-180, 180, step=5.625)

        x_dataset = x_dataset.reindex(latitude=new_lat, longitude=new_lon, method='nearest')

        return x_dataset


    def download(self):
        def get_last_batch(data_dir):
            already_downloaded_files = os.listdir(self.data_dir)
            already_downloaded_batches = [int(file.strip(".nc").split('_')[1]) for file in already_downloaded_files]
            return max(already_downloaded_batches)

        if self.continue_download:
            last_batch = get_last_batch(self.data_dir)
            next_batch = last_batch + 1
            i = next_batch*self.batch_size + self.start_time
            batch_num = next_batch
        else:
            i = self.start_time
            batch_num = 0

        print(f"Starting with sample num: {i}, batch: {batch_num}")

        with Progress() as progress:
            task_download = progress.add_task("[red]Downloading ocean wave data...", total=self.end_time-self.start_time)
            task_batch = progress.add_task("[red]Downloading batch...", total=self.batch_size)
            minibatch_list = [] 

            while i <= self.end_time:
                start = i
                end = min(i+self.minibatch_size-1, self.end_time)
                minibatch = self._download_minibatch(self.variables, start, end)
                progress.advance(task_download, advance=self.minibatch_size)
                progress.advance(task_batch, advance=self.minibatch_size)
                minibatch_list.append(minibatch)
                i += self.minibatch_size
                
                if len(minibatch_list)*self.minibatch_size >= self.batch_size:
                    batch_xarray = xr.concat(minibatch_list, dim = 'time')
                    batch_xarray.to_netcdf(os.path.join(self.data_dir, f'wave_{batch_num}.nc'))
                    batch_num += 1
                    minibatch_list = []
                    progress.reset(task_batch)

            batch_xarray = xr.concat(minibatch_list, dim = 'time')
            batch_xarray.to_netcdf(os.path.join(self.data_dir, f'wave_{batch_num}.nc'))
    