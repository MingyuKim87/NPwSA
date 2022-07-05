import os
import numpy as np
import torch
from .data_import import *

# periodic noise
import GPy

class movielens10k_dataset_loader(object):
    def __init__(self, datatype, user_file_path, rating_file_path, \
        item_file_path, num_task, noise_type=None, is_shuffle=True, is_torch=True,\
        trainingdata_ratio=0.8, drop_user_id=False):

        # parameters
        self.datatype = datatype
        self.num_task = num_task
        self.noise_type = noise_type
        self.noise_coef = 1e+3
        self.is_shuffle = is_shuffle
        self.user_file_path = user_file_path
        self.rating_file_path = rating_file_path
        self.item_file_path = item_file_path
        self.is_torch = is_torch
        self.trainingdata_ratio = trainingdata_ratio
        self.drop_user_id = drop_user_id

        # data
            # dataframe
        (self.x, self.y) = self._dataset_split()

        # Index (numpy array)
        self.user_info = self.x['user_id'].unique()

        # batch order
        self.user_order = self.set_batch_user_order() \
            if self.is_shuffle else self.user_info

        # raw data (numpy)
        self.x = self.x.to_numpy()
        self.y = self.y.to_numpy()

        # data
        self.data = self._set_batches()

        # length
        self.length = self.__len__()

    def __len__(self):
        length = len(self.user_info) // self.num_task
        return length

    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        if self.index >= self.length:
            # re-arrange batches
            self.user_order = self.set_batch_user_order() \
                if self.is_shuffle else self.user_info
            self.data = self._set_batches()

            # reset iteration
            self.index = 0
            raise StopIteration
        
        batch = self.data[self.index]
        self.index += 1
        return batch
            
    def _set_batches(self):
        '''
            make batches
        '''

        # container
        batches = []

        # user defined
        for i in range(0, len(self.user_order), self.num_task):
            # Treatement
            sampled_user_list = self.user_order[i:i+self.num_task]

            # Sample by user
            (x, y), min_count = self._sample_data_by_user(sampled_user_list)

            # torch tensor
            if self.is_torch:
                # type cast (numpy to tensor + 32bit float)
                x = torch.from_numpy(x).float()
                y = torch.from_numpy(y).float()

            # append
            batches.append((x,y,min_count))

        return batches

    def _periodic_noise(self, x_values, num_data_points):
        # kernel
        kernel = GPy.kern.src.periodic.PeriodicExponential(
                period=np.pi,
                n_freq=30)

        # covariance matrix
        C = kernel.K(x_values, x_values)

        # Sampling (GP prior)
        noise = np.random.multivariate_normal(np.zeros(num_data_points), C)[:, None]

        # Centering
        noise = (noise - noise.mean())

        return noise
            
    def _dataset_split(self):
        '''
            assign dataset
        '''

        train_set, val_set, test_set = \
            movielens_10k_data(self.user_file_path, self.rating_file_path, \
                self.item_file_path, self.trainingdata_ratio)

        if self.datatype == "train":
            (x, y) = train_set
        elif self.datatype == "val":
            (x, y) = val_set
        elif self.datatype == "test":
            (x, y) = test_set
        else:
            NotImplementedError

        return (x, y)

    def _sample_user_by_num_task(self):
        '''
            select users by num task

            Args:
                user_info : numpy array
        '''
    
        indices = np.random.permutation(len(self.user_info))
        return self.user_info[indices][:self.num_task]
        

    def _min_data_count_and_index_list(self, sampled_user_list, user_id_array):
        '''
            Figure out minimum data count
        '''
        # data count
        temp = []
        sample_indices = []
        for user_id in sampled_user_list:
            index = (user_id_array == user_id)

            # filter
            filtered_np = self.x[index, :]

            # len
            temp.append(len(filtered_np))

            # index matrix
            sample_indices.append(np.random.permutation(temp[-1]))

        # figure out "min data count"
        min_count = np.min(np.array(temp))

        # sample_indices
        sample_indices = [sample_index[:min_count] \
            for sample_index in sample_indices]

        return min_count, np.array(sample_indices)

    def _sample_data_by_user(self, sampled_user_list):
        '''
            sample data by user

            Args: 
                sampled_user_id : numpy

            Returns:
        '''
        # container
        container_x = []
        container_y = []
        
        # index array
        user_id_array = self.x[:,0] 

        # min data count
        min_data_count, index_matrix = \
            self._min_data_count_and_index_list(sampled_user_list, user_id_array)

        # sampling
        for user_id, index_list in zip(sampled_user_list, index_matrix):
            # index
            index = (user_id_array == user_id)

            # filter
            filtered_data_x = self.x[index, :] # (len(index), 45)
            filtered_data_y = self.y[index] # (len(index))

            # slicing
            x = filtered_data_x[index_list, :]
            y = filtered_data_y[index_list]

            # if drop user_id
            if self.drop_user_id:
                x = filtered_data_x[index_list, 1:]
            
            # matching ndim between filtered_data_x and filtered_data_y
            y = y[:, None]

            # noise 
            if self.noise_type == "periodic":
                noise = self._periodic_noise(x, len(index_list))
                y = y + self.noise_coef * noise
            
            # append
            container_x.append(x)
            container_y.append(y)

        return (np.array(container_x), np.array(container_y)), min_data_count

    def set_batch_user_order(self):
        perm_index = np.random.permutation(len(self.user_info))
        
        user_order = self.user_info[perm_index]
        # self.user_order = np.random.permutation(len(self.user_info))
        return user_order

if __name__ == "__main__":
    
    # file path
    user_file_path = "./data/movielens/ml-100k/u.user"
    rating_file_path =  "./data/movielens/ml-100k/u.data"
    item_file_path =  "./data/movielens/ml-100k/u.item"

    # datatype
    datatype = "train"
    
    # dataset
    movielens10k_dataset_loader = movielens10k_dataset(datatype, user_file_path, rating_file_path, \
        item_file_path, 50)
    
    print("-"*10, "first iteration", "-"*10)
    for data in movielens10k_dataset_loader:
        x, y, count = data
        print(x.shape)

    print("-"*10, "second iteration", "-"*10)
    for data in movielens10k_dataset_loader:
        x, y, count = data
        print(x.shape)

    

    

    


    

    
    

    


