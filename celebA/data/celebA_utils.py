import math
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn_image as isns

# declare 
irange = range

def img_mask_to_np_input(img, mask, normalize=True):
    """
    Given an image and a mask, return x and y tensors expected by Neural
    Process. Specifically, x will contain indices of unmasked points, e.g.
    [[1, 0], [23, 14], [24, 19]] and y will contain the corresponding pixel
    intensities, e.g. [[0.2], [0.73], [0.12]] for grayscale or
    [[0.82, 0.71, 0.5], [0.42, 0.33, 0.81], [0.21, 0.23, 0.32]] for RGB.

    Parameters
    ----------
    img : torch.Tensor
        Shape (N, C, H, W). Pixel intensities should be in [0, 1]

    mask : torch.ByteTensor
        Binary matrix where 0 corresponds to masked pixel and 1 to a visible
        pixel. Shape (N, H, W). Note the number of unmasked pixels must be the
        SAME for every mask in batch.

    normalize : bool
        If true normalizes pixel locations x to [-1, 1] and pixel intensities to
        [-0.5, 0.5]

    Returns
    ------------
        x : unoccluded pixel positions
        y : masked img
    """

    #task_size == batch_size
    task_size, num_channels, height, width = img.size()

    # Create a mask which matches exactly with image size which will be used to extract pixel intensities. 
        # Accoding to tasks, there are different mask matrices. 
    mask_img_size = mask.unsqueeze(dim=1).repeat(1, num_channels, 1, 1)

    # The number of points corresponds to the number of visible pixels in mask, 
        # torch.nonzero() -> 0이 아닌 값을 가진 값들의 index를 모두 도출하도록 함(무조건 2D tensor로 나오지만 마지막 차원의 vector는 tensor의 dimension을 따르게 되어있다. )
    num_points = mask[0].nonzero().size(0) #유효한 height와 width를 모두 더한 값이 됨

    # Index : mask [task_size, height, width]
    nonzero_idx = mask.nonzero() #[num_nonzeros(task_size * num_points), 3(task, height, width)]

    x = nonzero_idx[:, 1:].view(task_size, num_points, 2).float() #[task_size, num_points, height, width]

    y = img[mask_img_size].view(task_size, num_channels, num_points) # (height, width) --> flatten (num_points)

    # Ensure correct shape, i.e. (task_size, num_points, num_channels)
    y = y.permute(0, 2, 1)

    if normalize:
        # x \in [-1,1] 
        x = (x - float(height) / 2) / (float(height) / 2)
        # y \in [-.5, .5]
        y -= 0.5

    return x, y

def random_context_target_mask(img_size, num_context, num_extra_target, is_same_channel=False):
    '''
        Args:
            img_size : (1, 32, 32) grayscale / (3, 32, 32) for RGB
            num_context : int 
            num_extra_target : int

        Return :
            context_mask : tensor #[width, height] composed of {0, 1}
            target_mask : tensor #[width, height] composed of {0, 1}
    '''

    _, height, width = img_size

    # Sample integers without replacement between 0 and the total number of pixels. 
        # Test
    Idx = np.random.randint(low=0, high=height*width-1, size=num_context + num_extra_target)
    measurement_1 = np.arange(height * width)[Idx]

    # Sample integers without replacement between 0 and the total number of pixels. 
    measurement = np.random.choice(range(height * width), size=num_context + num_extra_target, replace=False)

    # Create mask containers
    context_mask = torch.zeros(width, height).byte() # Memory efficiency
    target_mask = torch.zeros(width, height).byte() # Memory efficiency

    # Update mask with measurement 
    for i, m in enumerate(measurement):
        row = int(m / width)
        col = m % width
        target_mask[row, col] = 1

        if i < num_context:
            context_mask[row, col] = 1

    return context_mask, target_mask


def batch_context_target_mask(img_size, num_context, num_extra_target, batch_size, repeat=False):
    '''
        Args:
            img_size : 
            num_context : 
            num_extra_target : 
            batch_size = task_size
            repreat : 동일한 mask가 모든 task에 동일하게 적용됨

        Returns:
        
    '''
        
    # Creat mask containers
        # *list -> 열거형으로 각 component들이 적혀있다. 따라서 함수의 component로 넣기에 매우 좋다.
        # list(1,2,3,4,5) 
    context_mask_batch = torch.zeros(batch_size, *img_size[1:]).byte()
    target_mask_batch = torch.zeros(batch_size, *img_size[1:]).byte()

    if repeat :
        context_mask, target_mask = random_context_target_mask(img_size, num_context, num_extra_target)

        for i in range(batch_size):
            context_mask_batch[i] = context_mask
            target_mask_batch[i] = target_mask

    else:
        for i in range(batch_size):
            context_mask, target_mask = random_context_target_mask(img_size, num_context, num_extra_target)

            context_mask_batch[i] = context_mask
            target_mask_batch[i] = target_mask

    return context_mask_batch, target_mask_batch


def xy_to_img(x, y, img_size, is_normalize=True):
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
    if is_normalize:
        x = x * float(height /2 ) + float(height / 2)
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


def inpaint(model, img, context_mask, device):
    """
    Given an image and a set of context points, the model samples pixel
    intensities for the remaining pixels in the image.

    Parameters
    ----------
    model : models.NeuralProcessImg instance

    img : torch.Tensor
        Shape (channels, height, width)

    context_mask : torch.Tensor
        Binary tensor where 1 corresponds to a visible pixel and 0 to an
        occluded pixel. Shape (height, width). Must have dtype=torch.uint8
        or similar. 

    device : torch.device
    """
    
    is_training = model.neural_process.training

    # For inpainting, use Neural Process in prediction mode
    is_train = False
    
    target_mask = 1 - context_mask  # All pixels which are not in context
    # Add a batch dimension to tensors and move to GPU

    img_batch = img.unsqueeze(0)
    context_batch = context_mask.unsqueeze(0)
    target_batch = target_mask.unsqueeze(0)
    
    p_y_pred = model(img_batch, context_batch, target_batch)
    # Transform Neural Process output back to image
    x_target, _ = img_mask_to_np_input(img_batch, target_batch)
    # Use the mean (i.e. loc) parameter of normal distribution as predictions
    # for y_target
    img_rec = xy_to_img(x_target.cpu(), p_y_pred.loc.detach().cpu(), img.size())
    img_rec = img_rec[0]  # Remove batch dimension
    # Add context points back to image
    context_mask_img = context_mask.unsqueeze(0).repeat(3, 1, 1)
    img_rec[context_mask_img] = img[context_mask_img]
    # Reset model to mode it was in before inpainting
    model.neural_process.training = is_training
    return img_rec



# plotting for celeb imgs
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
    plt.imshow(img_grid.permute(1, 2,0).numpy())

    # Save the file
    if is_save:
        plt.savefig(FILE_NAME, dpi=300)
    else:
        plt.show()

    plt.clf()

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
        g = isns.imghist(img.permute(1,2,0)) 

        # ticks and scale bar
        cax = plt.gcf().axes[1]
        isns.scientific_ticks(cax)
    else:
        g = isns.imgplot(img.permute(1,2,0), cbar=False) 

    # Save the file
    if is_save:
        plt.savefig(FILE_NAME, dpi=300)
    else:
        plt.show()

    plt.clf()



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









    