import numpy as np
import torch
import GPy
from math import pi
from torch.utils.data import Dataset, DataLoader


# visually valid 
from .helper.plot import plot_test

class SineData(Dataset):
    """
    Dataset of functions f(x) = a * sin(x - b) where a and b are randomly
    sampled. The function is evaluated from -pi to pi.

    Parameters
    ----------
    amplitude_range : tuple of float
        Defines the range from which the amplitude (i.e. a) of the sine function
        is sampled.

    shift_range : tuple of float
        Defines the range from which the shift (i.e. b) of the sine function is
        sampled.

    num_tasks : int
        Number of samples of the function contained in dataset.

    num_points : int
        Number of points at which to evaluate f(x) for x in [-pi, pi].
    """
    def __init__(self, amplitude_range=(-4., 4.), shift_range=(-.5, .5), x_range=(-np.pi, np.pi),
                 num_tasks=1000, num_points=100, x_dim=1, y_dim=1, is_noise=False):
        
        # Parameters
        self._num_tasks = num_tasks
        self._num_points = num_points
        self.amplitude_range = amplitude_range
        self.shift_range = shift_range
        self.x_range = x_range

        self.x_dim = 1  # x and y dim are fixed for this dataset.
        self.y_dim = 1

        # is noise
        self._is_noise = is_noise

        # Generate data
        self.data = self.generate_data()
        

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self._num_tasks

    def __set_periodic_kernel(self, n_freq=30):
        kernel = GPy.kern.src.periodic.PeriodicExponential(
                period=np.pi,
                n_freq=n_freq
            )
        return kernel

    def __generate_sine_functions(self):
        a_min, a_max = self.amplitude_range
        b_min, b_max = self.shift_range

        # Sample random amplitude
        a = (a_max - a_min) * np.random.rand() + a_min

        # Sample random shift
        b = (b_max - b_min) * np.random.rand() + b_min
    
        # Shape (num_points, x_dim)
        x = np.linspace(self.x_range[0], self.x_range[1], self._num_points)
        x = np.expand_dims(x, axis=-1)

        # Shape (num_points, y_dim)
        y = a * np.sin(x - b)

        return x.astype(np.float32), y.astype(np.float32), a

    def __generate_periodic_functions(self, n_freq=30):
        kernel = self.__set_periodic_kernel()

        # x_values
        x_values = np.linspace(self.x_range[0], self.x_range[1], self._num_points)
        x_values = np.expand_dims(x_values, axis=-1)

        # Convariance Matrix
        C = kernel.K(x_values, x_values).astype(np.float32) + (1e-4 * np.eye(self._num_points))

        # Sampling
        y = np.random.multivariate_normal(np.zeros(self._num_points), C)[:, None]
        y = (y - y.mean())

        return x_values.astype(np.float32), y.astype(np.float32)

    def __generate_functions(self, is_noise=False):
        # sine func
        x_sin, y_sin, a = self.__generate_sine_functions()
        # noise func
        x_gp, y_gp = self.__generate_periodic_functions()

        if is_noise:
            y = y_sin + (a * y_gp)
        else:
            y = y_sin

        return x_sin, y

    def generate_data(self):
        data = []

        for i in range(self._num_tasks):
            x, y = self.__generate_functions(self._is_noise)
            data.append((x,y))

        return data

    
if __name__ == '__main__':
    dataset = SineData(num_tasks=100, num_points=100, x_range=(-5,5), is_noise=True)

    x, y = dataset[0]

    context_index = np.random.randint(low=0, high=100, size=5)

    context_x = x[context_index]
    context_y = y[context_index]
    
    print(x.shape)
    print(y.shape)

    plot_test(context_x, context_y, x, y)
