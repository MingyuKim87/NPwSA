import os
import glob #사용자가 제시한 조건의 
import numpy as np
import torch
from math import pi
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

import pandas

from .celebA_utils import *

class CelebADataset(Dataset):
    '''
        CelebA dataset. 
        PIL packages can load the image as data type "H x W x C"
    '''

    def __init__(self, datatype, celebA_root_path, num_point, img_size, \
        transform=None, attr=None, is_rgb=True):
        """
        Args :
            celebA_root_path : root path 
            datatype : train loader or test loader
            num_point : total data point in each task
            img_size : img_size (int, square imgs)
            transform : transformation function by torchvision
            attr : what class you select
            is_rgb : gray scale or color scale
        """
        # parameters
        self.root_path = celebA_root_path
        self.img_size = img_size
        self.channel, self.height, self.width = img_size
        self.num_point = num_point
        self.total_points = self.height*self.width # square image
        self.datatype = datatype
        self.attr = attr
        self.is_rgb = is_rgb

        # image paths
        self.path_list = self._get_image_path_list()

        # counting all elements
        self.total_count = len(self.path_list)
        self.train_count = int(round(self.total_count * 0.8))
        self.test_count = self.total_count - self.train_count
        self.dataset_lenght = self.train_count \
            if ((self.datatype == 'train') or (self.datatype == 'test')) \
            else self.test_count
        
        # split train and test
        if self.datatype == 'train' :
            self.img_paths = self.path_list[:self.train_count]
        elif self.datatype == 'test':
            self.img_paths = self.path_list[:self.train_count]
            self.num_point = self.total_points # for valiation (visualization)
        elif (self.datatype == 'val'):
            self.img_paths = self.path_list[self.train_count:]
            self.num_point = self.total_points # for valiation (visualization)
        else:
            NotImplementedError
            
        # transform function
        self.transform = transform

        # load 
        self.images = self._load_images()
        self.masks, self.xs = self._generate_mask_matrix()

        # data
        self.data = self.generate_data()

    def __len__(self):
        if self.datatype == 'train':
            return self.train_count
        elif (self.datatype == 'val') or (self.datatype == 'test'):
            return self.test_count
        else:
            NotImplementedError

    def __getitem__(self, idx):
        return self.data[idx]
    
    def _load_images(self):
        '''
            load all images of celebA

            Return :
                list of all images (torch)
        '''
        # container
        container = []
        
        for path in self.img_paths:
            img = Image.open(path)

            if self.transform:
                img = self.transform(img)

                # transpose : C X H X W --> H x W x C
                img = torch.transpose(img, 0, 2)

                container.append(img)

        return container
    
    def _generate_mask_matrix(self):
        '''
            load mask matrices 

            Returns:
                list of mask matrix and dimensional matrix
        '''
        # container
        mask_container = []
        x_container = []
        
        for path in self.img_paths:
            mask, x = self._random_mask_matrix()

            # rbf or gray scale
            # mask = mask.unsqueeze(dim=0).repeat(3, 1, 1) if self.is_rgb \
            #     else mask.unsqueeze(dim=0)

            # append
            mask_container.append(mask)
            x_container.append(x)

        return mask_container, x_container

    def get_img_size(self):
        sample, _ = self.__getitem__(0)
        return sample.size()

    def _get_image_path_list(self):
        # image dir path
        img_dir_path = os.path.join(self.root_path, "celebA_img")

        # csv file and pandas dataframe
        attr_celebA = os.path.join(self.root_path, "list_attr_celeba.csv")
        df = pandas.read_csv(attr_celebA)

        # if self.attr exists
        if self.attr is not None:
            mask = (df[self.attr] == 1)
            filtered_df = df.loc[mask, :]
            image_list = filtered_df['image_id'].to_numpy()
        else:
            image_list = df['image_id'].to_numpy()

        # Re-arrange lists
        image_list = [os.path.join(img_dir_path, img_path) for img_path in image_list]
        
        return image_list

    def _random_mask_matrix(self):
        '''
            Generate random mask matrix

            Returns:
                mask : same dimensional matrix as img (one image)
                x : position list corresponding to the mask matrix (one image)
        '''
        # container
        mask = torch.zeros(self.total_points)

        # sample k data points
        index = torch.multinomial(torch.rand(self.total_points), self.num_point, replacement=False)

        # indicate selected componets
        mask[index] = 1

        # reshape
        mask = mask.view(self.height, self.width).long()

        # index
        x = torch.nonzero(mask)

        return mask, x

    
    def generate_data(self, is_normalize=True, is_rgb=True):
        '''

        '''
        
        # container
        container = []

        for idx in range(self.dataset_lenght):
            # select
            img = self.images[idx]
            mask = self.masks[idx]

            extended_mask = mask.unsqueeze(dim=-1).repeat(1, 1, 3).bool() if is_rgb \
                else mask.unsqueeze(dim=-1).bool()

            x = self.xs[idx]

            # for images (channel)
                # select pixels H X W X C --> [all points]
            y = img[extended_mask]
            y = y.view(-1, 3)

            if is_normalize:
                # x \in [-1,1] 
                x = (x - float(self.height) / 2) / (float(self.height) / 2)
                # y \in [-.5, .5]
                y -= 0.5

            # append
            container.append((mask, x, y))
        
        return container

    def re_generate_mask_and_data(self):
        '''
            explicitly call generating random mask matrix for randomized training
        '''
        self.masks, self.xs = self._generate_mask_matrix()
        self.data = self.generate_data()

def context_target_split_trainer(x, y, num_context, num_total_point, is_test=False, **kargs):
    """
        Args:
            x : batch_size, total_data_points, x_dim
            y : batch_size, total_data_points, y_dim
            num_context : scalar (int)
            num_extra_target : scalar (int)
            is_test : use all or sample

        Returns:
            x_context, y_context, x_target, y_target
    """

    num_points = x.shape[1]
    
    # Sample locations of context and target points (for meta-train)
    locations = np.random.choice(num_points,
                                 size=num_total_point,
                                 replace=False)

    x_context = x[:, locations[:num_context], :]
    y_context = y[:, locations[:num_context], :]
    x_target = x[:, locations, :] if not is_test else x
    y_target = y[:, locations, :] if not is_test else y         

    return x_context, y_context, x_target, y_target, locations[:num_context]

def merge_context_target_image(context_x, context_y, target_x, target_y, \
    img_size, is_normalize=True):

    channel, height, width = img_size
    batch_size, _, _ = context_x.size() # x is torch.Tensor
    
    
    if is_normalize:
        # x
        context_x = context_x * float(height/2) + float(height/2)
        target_x = target_x * float(height/2) + float(height/2)
        # y
        context_y += 0.5
        target_y += 0.5

    # type cast : float --> long
    context_x = context_x.long()
    target_x = target_x.long()
    
    # size
    new_img_size = tuple((batch_size,)) + tuple((height, width, channel))
    img = torch.zeros(new_img_size)

    for i in range(batch_size):
        img[i, target_x[i, :, 0], target_x[i, :, 1], :] = target_y[i, :, :]
        # img[i, context_x[i, :, 0], context_x[i, :, 1], :] = context_y[i, :, :]

    # H X W X C --> C X H X W
    img = torch.transpose(img, 3, 1)
    
    return img


if __name__ == '__main__':        
    # DEBUG
    img = isns.load_image("polymer") * 1e-9

    print(img)
    
    
    # Parameters
    img_size = [3, 32, 32]
    channel, height, width = img_size
    batch_size = 32

    # Transform
    transform = transforms.Compose([
        # transforms.CenterCrop(89),
        transforms.CenterCrop(95),
        transforms.Resize(height),
        transforms.ToTensor()
    ])

    path = '/home/mgyukim/Data/celebA/'
    dataset = CelebADataset("test", path, img_size=img_size, num_point=100, transform=transform, attr='Bald')
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for i, data in enumerate(dataloader):
        mask, x, y = data
        
        # parameters
        num_context = 100
        num_extra_target = 100
        num_total_datapoints = 1024

        # img size
        img_size = (3, 32, 32)
        
        # context, target
        # num_context, num_target
        num_context = np.random.randint(low=num_context, high=(num_total_datapoints - (num_context + num_extra_target)))
        num_extra_target = np.random.randint(low=num_extra_target, high=(num_total_datapoints - num_context))
        num_total_point = num_total_datapoints

        # Split context and target
        context_x, context_y, target_x, target_y, _ = \
            context_target_split_trainer(
                x = x, 
                y = y,
                num_context = num_context,
                num_total_point = num_total_point,
            )
        
        break

    # trasnlate into IMGs.
    predict_img = xy_to_img(target_x, target_y, img_size)
    context_img = xy_to_img(context_x, context_y, img_size)

    # combined_img_true = merge_context_target_image(context_x, context_y, target_x, target_y, img_size)

    # plotting
    plot_celebA_img(predict_img[0], True, file_name='/celebA_predict_img_{}.png'.format('target'))
    plot_celebA_img(context_img[0], True, file_name='/celebA_context_img_{}.png'.format('context'))
    # plot_celebA_img(combined_img_true, True, file_name='/celebA_combined_img_{}.png'.format('true'))
    

    '''
    # Restore images
    img = torch.zeros(32,32,32,3)
    for i, (position, rgbs) in enumerate(zip(x, y)):
        # Unnormalize x and y
        position = position * float(height / 2) + float(height / 2)
        position = position.long()
        rgbs += 0.5
        
        print(position[:,0].shape)
        print(position[:,1].shape)
        print(img[i, position[:, 0].long(), position[:, 1].long(), : ].shape)
        print(rgbs.shape)

        img[i, position[:, 0].long(), position[:, 1].long(), :] = rgbs[:, :]

        
    img = torch.transpose(img, 3, 1)

    print(img.shape)
    plot_celebA_img(img, True, './Result')
    '''
    

    
    