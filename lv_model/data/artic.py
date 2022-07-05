import os
from functools import reduce
import numpy as np
import GPy
import matplotlib.pyplot as plt
from attrdict import AttrDict

import torch
from torch.utils.data import Dataset, DataLoader

# visually valid 
from .helper.plot import plot_test

class ArticData(Dataset):
    def __init__(self, file_paths, num_tasks, is_test=False, **kargs):
        """
            Args:
                file_path : file_path
        """
        self._file_paths = file_paths
        self._num_tasks = num_tasks
        self._num_points = None
        self._is_test = is_test

        self._x, self._y = self.__load_global_stat(self._file_paths[0], self._file_paths[1])
        self.data = self.generate_data()

    def __len__(self):
        return self._num_tasks

    def __getitem__(self, index):
        return self.data[index]

    def __load_global_stat(self, global_file_path, outlier_file_path):
        '''
		    Args:
			    e.g)
			    outliers = np.loadtxt('./Outliers.csv', delimiter=',', dtype=np.float32, skiprows=1)
			    global_file_path = './GlobalInformation.csv'
	    '''
	
        outliers = np.loadtxt(outlier_file_path, delimiter=',', dtype=np.float32, skiprows=1)
        global_SIC = np.loadtxt(global_file_path, delimiter=',', dtype=np.float32, skiprows=1)

        # Make x values (time)
        x = np.array(np.arange(0, global_SIC.shape[0]))

        # Filter outliers
        x = np.delete(x, np.squeeze(outliers).astype(np.int32), 0)
        y = np.delete(global_SIC, np.squeeze(outliers).astype(np.int32), 0)

        # Slicing y values
        y = y[:,4:]

        # continous : copy numpy array to torch.Tensor 
        x, y = torch.from_numpy(x), torch.from_numpy(y)
        x, y = x.contiguous(), y.contiguous()

        return x, y

    def generate_data(self):
        # y index list
        y_idx_list = torch.randint(low=0, high=self._y.shape[-1] -1, size=2)
        
        data = []

        for i in range(self._num_tasks):
            x, y = \
                self._x.float(), self._y.float()

            if self._is_test:
                # idx_list
                idx_list = torch.linspace(x.shape[0]-365, x.shape[0]-1, step=int(x.shape[0]-365 / 7), dtype=torch.long)

                # slicing
                x, y = x[idx_list], y[idx_list, y_idx_list]
            
            # set dataset._num_points
            if self._num_points is None:
                self._num_points = y.shape[-2]
        
            data.append((x,y))

        return data

