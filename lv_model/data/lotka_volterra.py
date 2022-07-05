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

class LVdata(Dataset):
    def __init__(self, file_path, **kargs):
        '''
        Args:
            file_path : file_path
        '''
        self._file_path = file_path
        self._is_noise = kargs.get("is_noise", None)
        self._noise_coef = kargs.get("noise_coef", 1)
        self._batches = self.generate_data()
        
        

    def __len__(self):
        return len(self._batches)

    def __getitem__(self, index):
        return self._batches[index]

    def _make_noise(self, kernel, x, y):
        # container
        noise_container = []
        
        # num_points
        num_points = y.shape[1]

        # noise
        for i in range(y.shape[0]):
            # allocation
            x_i_np = x[i].numpy()
            
            
            C = kernel.K(x_i_np, x_i_np).astype(np.float32) + (1e-4 * np.eye(num_points))
            
            
            noise_list = \
                [np.random.multivariate_normal(np.zeros(num_points), C)[:, None] \
                    for _ in range(y.shape[-1])]

            # np stack
            noise = np.concatenate(tuple(noise_list), axis=-1)

            noise_container.append(noise)

        noise = np.stack(noise_container, axis=0)

        return noise

    def generate_data(self):
        # load dataset
        data = torch.load(self._file_path)

        if self._is_noise:
            for i, batch_data in enumerate(data):

                # filtering 
                x = batch_data['x']
                y = batch_data['y']

                xc = batch_data['xc']
                yc = batch_data['yc']

                xt = batch_data['xt']
                yt = batch_data['yt']

                # get shape
                num_total_points = x.shape[1]
                num_context_points = xc.shape[1]
                num_target_points = xt.shape[1]

                # kernel
                kernel = GPy.kern.src.periodic.PeriodicExponential(
                    period=np.pi,
                    n_freq=30)

                # make a noise
                noise = self._make_noise(kernel, x, y)
                    
                # add noise
                y = y + (self._noise_coef * torch.from_numpy(noise).float())

                # repackaging
                    # indexing
                locations = np.random.choice(num_total_points,
                                size=num_total_points,
                                replace=False)

                # reassigning
                data[i]['x'] = x
                data[i]['y'] = y

                data[i]['xc'] = x[:, locations[:num_context_points], :]
                data[i]['yc'] = y[:, locations[:num_context_points], :]
            
                data[i]['xt'] = x[:, locations[num_context_points:], :]
                data[i]['yt'] = y[:, locations[num_context_points:], :]

        return data


class HLdata(Dataset):
    def __init__(self, file_path, num_tasks, **kargs):
        """
            Args:
                file_path : file_path
        """
        self._file_path = file_path
        self._num_tasks = num_tasks
        self._num_points = None

        self._times, self._pops = self.__load_hare_lynx_data()
        self.data = self.generate_data()

    def __len__(self):
        return self._num_tasks

    def __getitem__(self, index):
        return self.data[index]

    def __load_hare_lynx_data(self):
        tb = np.loadtxt(self._file_path)
        
        # Set variables
            # times : [num_data, 1]
        times = np.expand_dims(tb[:, 0], axis=-1)
            # times : [num_data, 2 (predetor, prey)]
        pops = np.stack((tb[:, 2], tb[:, 1]), axis=-1)

        return times, pops

    def generate_data(self):
        data = []

        for i in range(self._num_tasks):
            x, y = \
                self._times.astype(np.float32), self._pops.astype(np.float32)

            # set dataset._num_points
            if self._num_points is None:
                self._num_points = y.shape[-2]
        
            data.append((x,y))

        return data

    





if __name__ == "__main__":
    dataset = LVdata("./data/dataset/train.tar", is_noise=True, noise_coef=100)
    dataset2 = HLdata("./data/dataset/LynxHare.txt", num_tasks=50)

    data = dataset[0]
    data2 = dataset2[0]

    context_index = np.random.randint(low=0, high=50, size=15)

    # slicing dataset (LV)
    x = data['x']
    y = data['y']

    xc = data['xc']
    yc = data['yc']

    xt = data['xt']
    yt = data['yt']

    # slicing dataset (HL)
    x2, y2 = data2
    context_x, context_y = x2[context_index], y2[context_index]

    # plotting
    plot_test(np.squeeze(xc[0,:,:]), yc[0,:,0], np.squeeze(x[0,:,:]), y[0,:,0], file_name="/LV_data_1.png")
    plot_test(np.squeeze(xc[0,:,:]), yc[0,:,1], np.squeeze(x[0,:,:]), y[0,:,1], file_name="/LV_data_2.png")

    plot_test(context_x, context_y[:,0], x2, y2[:,0], file_name="/HL_data_1.png")
    plot_test(context_x, context_y[:,1], x2, y2[:,1], file_name="/HL_data_2.png")



    
    
    




    

        