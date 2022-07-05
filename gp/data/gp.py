
from functools import reduce
import numpy as np
import GPy
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader

# visually valid 
from .helper.plot import plot_test

class GPData(Dataset):
    def __init__(self, num_tasks, num_points, x_dim=1, y_dim=1, sigma_scale=3., l1_scale=3., n_freq=30, **kargs):
        '''
        Args:
            num_samples : length of dataset (total interation on a batch = length of dataset / task_size)
            num_points : the number of points for plotting
            x_dim : integer >= 1, dimension of input points
            y_dim : integer >= 1, dimension of output points
            l1_scale : float, scaling value for rbf kernel 
            sigma_scale : float, scaling for variance w.r.t rbf kernel
            noise_std : float, noisy std for the rbf kernel
            n_freq : int, frequancy for the periodic kernel
        '''

        self._num_tasks = num_tasks
        self._num_points = num_points

        self._x_dim = x_dim
        self._y_dim = y_dim

        self._n_freq = n_freq

        self._x_minimum = kargs.get("x_minimum", -4)
        self._x_maximum = kargs.get("x_maximum", 4)

        # Amplitude shift
        self._shift = kargs.get("shift", None)

        # whether RBF kernel parameters are random
        self._is_random = kargs.get("is_random", False)

        # t noise (come from bootstrapped NPs)
        self._is_t_noise = kargs.get("is_t_noise", False)

        # K matrix
        self.kernel_list = self.__set_kernels(\
            sigma_scale=sigma_scale, \
            l1_scale=l1_scale,\
            n_freq=n_freq, \
            is_random=self._is_random
            )

        # Genereate Data
        self.data = self.generate_data()
        
    def __len__(self):
        return self._num_tasks

    def __getitem__(self, index):
        return self.data[index]

    def __set_kernels(self, **kargs):        
        kernel_list = []

        # For rbf kernel
        if self._is_random:
            sigma_scale = 0.1 + (np.random.rand() * 0.9)
            l1_scale = 0.1 + (np.random.rand() * 0.5)
            
            kernel = GPy.kern.RBF(input_dim=self._x_dim,\
                variance=sigma_scale,\
                lengthscale=l1_scale
            )

        elif kargs.get("sigma_scale", None) is not None and \
            kargs.get("l1_scale", None) is not None:

            kernel = GPy.kern.RBF(input_dim=self._x_dim,\
                    variance=kargs.get("sigma_scale"),\
                    lengthscale=kargs.get("l1_scale")
                )
        else:
            NotImplementedError
    
        kernel_list.append(kernel)


        # For periodic kernel (noise)
        if kargs.get("n_freq", None) is not None and self._is_t_noise != True:
            kernel = GPy.kern.src.periodic.PeriodicExponential(
                period=np.pi,
                n_freq=kargs.get("n_freq")
            )
            kernel_list.append(kernel)
        
        # exceptional
        kernel_list = None if len(kernel_list)==0 else kernel_list

        return kernel_list
        
            
    def __get_covariance_matrix(self, x_values):
        C_list = [kernel.K(x_values, x_values).astype(np.float32) for kernel in self.kernel_list]

        # sum of all list : reduce functiools
        C = reduce(lambda x, y : x + y, C_list)

        return C

    def _generate_functions(self):
        num_total_points = self._num_points
        range_x_value = np.linspace(self._x_minimum, self._x_maximum, num=num_total_points) #[the number of total points, ] 

        # x_values
        x_values = np.expand_dims(range_x_value, axis=-1) # [num_total_points, 1] # this function only work well "x_size = 1"

        # Convariance Matrix
        C = self.__get_covariance_matrix(x_values) + (1e-4 * np.eye(num_total_points))

        # Sampling
        y = np.random.multivariate_normal(np.zeros(self._num_points), C)[:, None]
        y = (y - y.mean())

        y = y + self._shift if self._shift is not None else y

        if self._is_t_noise and self._n_freq is None:
            t_noise = 0.15 * np.random.standard_t(2.1, y.shape)
            y += t_noise

        return x_values.astype(np.float32), y.astype(np.float32)

    def generate_data(self):
        data = []

        for i in range(self._num_tasks):
            x, y = self._generate_functions()
            data.append((x,y))

        return data

    def update_data(self):
        """
            Set function 
        """
        
        # K matrix
        self.kernel_list = self.__set_kernels()

        # Genereate Data
        self.data = self.generate_data()

    
if __name__ == '__main__':
    a = np.random.rand()

    print(a)