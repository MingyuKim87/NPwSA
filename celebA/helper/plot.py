import math
import os
import numpy as np
import torch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import matplotlib.ticker as mticker
import seaborn_image as isns

import pandas as pd
import seaborn as sns

irange = range

def plot_1D_regression(context_x, context_y, target_x, target_y, \
    idx = 0, *args, **kargs):
    '''
        Plots the predictive distribution (mean, var) and context points

        Args:
            All arguments are "NP arrays"
            target_x : [batch_size, the number of total_points, x_size(dimension)] 
            target_y : [batch_size, the number of total_points, y_size(dimension)] 
            context_x : [batch_size, the number of context_points, x_size(dimension)] 
            context_y : [batch_size, the number of context_points, y_size(dimension)] 
            pred_y : [batch_size, the number of total_point, y_size(dimension)] same as target_y
            var  : [batch_size, the number of total_point, y_size(dimension)] same as var
    '''
    # Path
    result_path = kargs.get("result_path", "./")
    file_name = kargs.get("file_name", "regression_result.png")
    FILE_NAME = os.path.join(result_path, file_name)

    # Pred
    pred = kargs.get("pred_mu", None)
    std = kargs.get("pred_sigma", None)

    
    if target_x.shape[0] == 1:
        context_x = np.squeeze(context_x[idx], axis=-1)
        context_y = np.squeeze(context_y[idx], axis=-1)
        target_x = np.squeeze(target_x[idx], axis=-1)
        target_y = np.squeeze(target_y[idx], axis=-1)
        pred = np.squeeze(pred[idx], axis=-1)
        std = np.squeeze(std[idx], axis=-1)
    else:
        context_x = np.squeeze(context_x[idx])
        context_y = np.squeeze(context_y[idx])
        target_x = np.squeeze(target_x[idx])
        target_y = np.squeeze(target_y[idx])
        pred = np.squeeze(pred[idx])
        std = np.squeeze(std[idx])
        
    # scatter plot
    plt.plot(context_x, context_y, 'ko', markersize=5)
    
    
    # Line plot
    plt.plot(target_x, target_y, 'k:', linewidth=1)
    if pred is not None and std is not None:
        # line plot
        plt.plot(target_x, pred, 'b', linewidth=2)

        # var
        plt.fill_between(target_x, \
            pred - std,\
            pred + std,
            alpha = 0.2,
            facecolor='#65c9f7',
            interpolate=True
        )
        
    # Make a plot pretty
    TITLE = kargs.get("title", "1D regression plot")
    ITERATION = kargs.get("iteration", None)

    # Title
    plt.title(TITLE, fontweight='bold', loc='center')
    
    # Label
    plt.xlabel("{} iterations".format(ITERATION)) if not ITERATION is None else None
    
    # Axis
    plt.yticks([-2, 0, 2], fontsize=16)
    plt.xticks([-4, 0, 4], fontsize=16)
    #plt.ylim([-1, 1])
    plt.grid(False)
    ax = plt.gca()
    ax.set_facecolor('white')

    # Save
    plt.savefig(FILE_NAME, dpi=300)

    # Close plt
    plt.clf()


def plot_attention_weights_heat_map(context_x, target_x, attention_weights, \
    idx=0, *args, **kargs):
    '''
        Plot heatap for attention weights

        Args:
            All arguments are "NP arrays"
            context_x : [batch_size, the number of context_points, x_size(dimension)] 
            target_x : [batch_size, the number of context_points, x_size(dimension)] 
            attention_weights_dict : [batch_size, the num of target set, the num of context set]
    '''
    # Formatting
    f = mticker.ScalarFormatter(useOffset=False, useMathText=True)
    g = lambda x,pos : "${}$".format(f._formatSciNotation('%1.10e' % x))
    
    
    # Path
    result_path = kargs.get("result_path", "./")
    file_name = kargs.get("file_name", "heatmap_result.png")
    FILE_NAME = os.path.join(result_path, file_name)

    # Make a plot pretty
    TITLE = kargs.get("title", "1D regression plot")
    ITERATION = kargs.get("iteration", None)
    
    # data pre-processing
    if target_x.shape[0] == 1:
        context_x = np.squeeze(context_x[idx], axis=-1)
        target_x = np.squeeze(target_x[idx], axis=-1)
        attention_weights = attention_weights[idx] if attention_weights is not None else None
    else:
        context_x = np.squeeze(context_x[idx])
        target_x = np.squeeze(target_x[idx])
        attention_weights = np.squeeze(attention_weights[idx]) if attention_weights is not None else None

    # Sorted 
    context_x_idx = sorted(range(context_x.shape[0]), key=lambda k : context_x[k])
    target_x_idx = sorted(range(target_x.shape[0]), key=lambda k : target_x[k])

    # rearrange the attention_weight
    context_x = context_x[context_x_idx]
    attention_weights = attention_weights[:, context_x_idx]

    # Make a figure
    fig, ax = plt.subplots()
    heatmap = ax.pcolor(attention_weights, cmap=plt.cm.Blues)

    # color-bar
    cbar = plt.colorbar(heatmap)

    # put the major ticks at the middle of each cell
    ax.set_xticks(np.arange(attention_weights.shape[1]) + 0.5, minor=False) # target
    ax.set_yticks(np.arange(attention_weights.shape[0]) + 0.5, minor=False) # input

    # extra columns row
    ax.set_xlim(0, int(attention_weights.shape[1])) # context
    ax.set_ylim(0, int(attention_weights.shape[0])) # target

    # want a more natural, table-like display
    ax.invert_yaxis()
    ax.xaxis.tick_top()

    # source words -> column labels
    ax.set_xticklabels(context_x, minor=False)
    ax.set_yticklabels(target_x, minor=False)
    
    # set ticks
    plt.xticks(rotation=45, fontsize=8)
    plt.yticks(rotation=25, fontsize=8)

    # formatting
    # ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
    # ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))

    # Title
    plt.title(TITLE, fontweight='bold', loc='center')

    # tight layout
    plt.tight_layout()
    
    # Save
    plt.savefig(FILE_NAME, dpi=300)

    # Close plt
    plt.clf()

def xy_to_img(x, y, img_size, normalize_type='xy'):
    '''
    Given an x and y returned by a neural processes, reconstruct images. 
    Missing pixels will have a value of 0.

    Args
    ---------
        x : torch.Tensor #[task_size, num_points, 2] <- containing normalized indices.  2 means (height, width)
            x \in [0,1] (normalized)
        y : torch.Tensor #[task_size, num_points, num_channels] where num_channel =1 is grayscale, num_channel = 3 is rgb
        img_size : tuple of int (1, 32, 32) / (3, 32, 32) <- (C, H, W)

    return 
    ---------
        img : 
    '''

    channel, height, width = img_size
    batch_size, _, _ = x.size() # x is torch.Tensor

    # Unnormalize x and y
    if normalize_type == 'xy':
        x = x * float(height /2 ) + float(height / 2)
        y += 0.5
    elif normalize_type == 'x':
        x = x * float(height /2 ) + float(height / 2)
    elif normalize_type == 'y':
        y += 0.5

    
    x = x.long()

    # Initialize empty image
    #img = torch.zeros((batch_size, ) + img_size)
    new_img_size = tuple((batch_size,)) + tuple((height, width, channel))
    img = torch.zeros(new_img_size)
    
    # x[i, :, 0], x[i, :, 1] index를 갖고 있다
    for i in range(batch_size):
        img[i, x[i, :, 0], x[i, :, 1], :] = y[i, :, :] # broadcasting 

    # H X W X C --> C X H X W
    img = torch.transpose(img, 3, 1)

    return img


def combine_context_target_image(context_x, context_y, target_x, target_y, task_size, img_size):
    _, height, width = img_size
    
    def unnormalize(x, y):
        # Unnormalize x and y
        x = x * float(height / 2) + float(height / 2)
        x = x.long()
        y += 0.5
        y = y.permute(0, 2, 1)

        return x, y

    context_mask, context_img = unnormalize(context_x, context_y)
    target_mask, target_img = unnormalize(target_x, target_y)

    img = torch.zeros((task_size,) + tuple(img_size))

    
    for i in range(task_size):
        img[i, :, target_mask[i, :, 0], target_mask[i, :, 1]] = target_img[i, :, :]
        img[i, :, context_mask[i, :, 0], context_mask[i, :, 1]] = context_img[i, :, :]
    
    return img

# plotting for celeb imgs
def plot_celebA_img(img, is_save = False, \
    result_path='./Result', file_name='/celebA_dataset.png', is_scale_bar=False):
    '''
        Args:
            img : C X H X W (one_img)

        Returns:
    '''
    # Path
    RESULT_PATH = result_path
    FILE_NAME = RESULT_PATH + file_name

    # setting
    isns.set_image(origin="upper")

    # Plotting
    if is_scale_bar:
        g = isns.imgplot(img.numpy()) 

        # ticks and scale bar
        # cax = plt.gcf().axes[1]
        # isns.scientific_ticks(cax)
    else:
        g = isns.imgplot(img.permute(1,2,0), cbar=False) 

    
    plt.savefig(FILE_NAME, dpi=300)
    plt.close()


# plotting for celeb A dataset
def plot_celebA_imgs(all_imgs, is_save = False, \
    result_path='./Result', file_name='/celebA_dataset.png'):
    '''
        Args:
            img : B X C X H X W (several_imgs)
    '''
    # Path
    RESULT_PATH = result_path
    FILE_NAME = RESULT_PATH + file_name

    # Img
    img_count = all_imgs.size(0)
    nrow = int(img_count / 2)

    # Plotting
    img_grid = make_grid(all_imgs, nrow=nrow, pad_value = 1.)

    # Visualize
    plt.imshow(img_grid.permute(1, 2,0).numpy(), origin='upper')
        
    # Save the file    
    plt.savefig(FILE_NAME, dpi=300)

    plt.close()


def make_grid(tensor, nrow=8, padding=2,
              normalize=False, range=None, scale_each=False, pad_value=0):
    """Make a grid of images.

    Args:
        tensor (Tensor or list): 4D mini-batch Tensor of shape (B x C x H x W)
            or a list of images all of the same size.
        nrow (int, optional): Number of images displayed in each row of the grid.
            The final grid size is ``(B / nrow, nrow)``. Default: ``8``.
        padding (int, optional): amount of padding. Default: ``2``.
        normalize (bool, optional): If True, shift the image to the range (0, 1),
            by the min and max values specified by :attr:`range`. Default: ``False``.
        range (tuple, optional): tuple (min, max) where min and max are numbers,
            then these numbers are used to normalize the image. By default, min and max
            are computed from the tensor.
        scale_each (bool, optional): If ``True``, scale each image in the batch of
            images separately rather than the (min, max) over all images. Default: ``False``.
        pad_value (float, optional): Value for the padded pixels. Default: ``0``.

    Example:
        See this notebook `here <https://gist.github.com/anonymous/bf16430f7750c023141c562f3e9f2a91>`_

    """
    if not (torch.is_tensor(tensor) or
            (isinstance(tensor, list) and all(torch.is_tensor(t) for t in tensor))):
        raise TypeError('tensor or list of tensors expected, got {}'.format(type(tensor)))

    # if list of tensors, convert to a 4D mini-batch Tensor
    if isinstance(tensor, list):
        tensor = torch.stack(tensor, dim=0)

    if tensor.dim() == 2:  # single image H x W
        tensor = tensor.unsqueeze(0)
    if tensor.dim() == 3:  # single image
        if tensor.size(0) == 1:  # if single-channel, convert to 3-channel
            tensor = torch.cat((tensor, tensor, tensor), 0)
        tensor = tensor.unsqueeze(0)

    if tensor.dim() == 4 and tensor.size(1) == 1:  # single-channel images
        tensor = torch.cat((tensor, tensor, tensor), 1)

    if normalize is True:
        tensor = tensor.clone()  # avoid modifying tensor in-place
        if range is not None:
            assert isinstance(range, tuple), \
                "range has to be a tuple (min, max) if specified. min and max are numbers"

        def norm_ip(img, min, max):
            img.clamp_(min=min, max=max)
            img.add_(-min).div_(max - min + 1e-5)

        def norm_range(t, range):
            if range is not None:
                norm_ip(t, range[0], range[1])
            else:
                norm_ip(t, float(t.min()), float(t.max()))

        if scale_each is True:
            for t in tensor:  # loop over mini-batch dimension
                norm_range(t, range)
        else:
            norm_range(tensor, range)

    if tensor.size(0) == 1:
        return tensor.squeeze(0)

    # make the mini-batch of images into a grid
    nmaps = tensor.size(0)
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height, width = int(tensor.size(2) + padding), int(tensor.size(3) + padding)
    
    # independent of devices (tensor)
    grid = tensor.new_full((3, height * ymaps + padding, width * xmaps + padding), pad_value)
    k = 0

    for y in irange(ymaps):
        for x in irange(xmaps):
            if k >= nmaps:
                break
            grid.narrow(1, y * height + padding, height - padding)\
                .narrow(2, x * width + padding, width - padding)\
                .copy_(tensor[k])
            k = k + 1

    return grid

def save_image(tensor, filename, nrow=8, padding=2,
               normalize=False, range=None, scale_each=False, pad_value=0):
    """Save a given Tensor into an image file.

    Args:
        tensor (Tensor or list): Image to be saved. If given a mini-batch tensor,
            saves the tensor as a grid of images by calling ``make_grid``.
        **kwargs: Other arguments are documented in ``make_grid``.
    """
    from PIL import Image
    grid = make_grid(tensor, nrow=nrow, padding=padding, pad_value=pad_value,
                     normalize=normalize, range=range, scale_each=scale_each)
    # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
    ndarr = grid.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    im = Image.fromarray(ndarr)
    im.save(filename)



def plot_training_curve(train_file_path, val_file_path=None, k=20, **kargs):
    '''
        Args:
            train_file_path : training result path
            val_file_path : val result path    
            k : paraemter for exponential moving average
    '''

    # Path
    result_path = kargs.get("result_path", "./")
    file_name = kargs.get("file_name", "training_curve.png")
    FILE_NAME = os.path.join(result_path, file_name)
    
    # Make a title
    TITLE = kargs.get("title", "training curve plot")

    # train
    train_temp = pd.read_csv(train_file_path, header=None)
    train_temp.columns = ["epochs", "loss", "negative_likelihood", "kld", "kld_in_attention"]
    train_temp['type'] = 'train'

    # add the EMA loss
    train_temp['EMA_loss'] = train_temp["loss"].ewm(k).mean()
    
    # val
    if val_file_path is not None:
        val_temp = pd.read_csv(val_file_path, header=None)
        val_temp.columns = ["epochs", "loss"]
        val_temp['type'] = 'val'

        # add the EMA loss
        val_temp['EMA_loss'] = val_temp["loss"].ewm(k).mean()
        
    # Make a plot pretty
    sns.set_style("darkgrid")
    
    # integrated dataframe
    integrated_df = pd.concat([train_temp, val_temp]) if val_file_path is not None else train_temp

    # plot a line 
    ax = sns.lineplot(data=integrated_df, x='epochs', y='loss', hue='type', alpha=0.3, legend=False)
    ax = sns.lineplot(data=integrated_df, x='epochs', y='EMA_loss', hue='type')

    # make pretty
        # Title
    plt.title(TITLE, fontweight='bold', loc='center')
    plt.tight_layout()
    
    # save fig
    plt.savefig(FILE_NAME, dpi=300)

    # Close plt
    plt.close()







    