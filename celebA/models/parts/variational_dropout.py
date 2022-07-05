import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from torch.optim import Optimizer

class VariationalDropout(nn.Module):
    def __init__(self, input_size, out_size, device, log_sigma2=-10, threshold=3):
        """
            Args:
                input_size: input size (int)
                out_size: output size (int)
                log_sigma2: Initial value of log sigma ^ 2. (float)
                    It is crusial for training since it determines initial value of alpha
                threshold: Value for thresholding of validation, for a mask layer
                    If log_alpha > threshold, then weight is zeroed
        """
        super(VariationalDropout, self).__init__()

        # device
        self.device = device
        
        self.input_size = input_size
        self.out_size = out_size

        self.theta = torch.nn.parameter.Parameter(torch.FloatTensor(input_size, out_size))
        self.bias = torch.nn.parameter.Parameter(torch.Tensor(out_size))

        self.log_sigma2 = torch.nn.parameter.Parameter(torch.FloatTensor(input_size, out_size).fill_(log_sigma2))

        self.reset_parameters()

        self.k = [0.63576, 1.87320, 1.48695]
        self.threshold = threshold

        # kl loss
        self.kl_loss = None

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_size)

        self.theta.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)

    @staticmethod
    def clip(input, to=8):
        input = input.masked_fill(input < -to, -to)
        input = input.masked_fill(input > to, to)

        return input

    def kld(self, log_alpha):

        first_term = self.k[0] * torch.sigmoid(self.k[1] + self.k[2] * log_alpha)
        second_term = 0.5 * torch.log(1 + torch.exp(-log_alpha))

        result = -(first_term - second_term - self.k[0]).sum() \
            / (self.input_size * self.out_size)

        return result


    def forward(self, input):
        """
        :param input: An float tensor with shape of [batch_size, input_size]
        :return: An float tensor with shape of [batch_size, out_size] and negative layer-kld estimation
        """

        # inputs size
        task_size, _, num_latents = input.size()

        # reshaping
        hidden = input.view((-1, num_latents))

        log_alpha = self.clip(self.log_sigma2 - torch.log(self.theta ** 2))
        kld = self.kld(log_alpha)

        if not self.training:
            mask = log_alpha > self.threshold
            hidden = torch.addmm(self.bias, hidden, self.theta.masked_fill(mask, 0))
            result = hidden.view((task_size, -1, num_latents))
            return result

        mu = torch.mm(hidden, self.theta)
        std = torch.sqrt(torch.mm(hidden ** 2, self.log_sigma2.exp()) + 1e-6)

        # rsample
        normal_dist = torch.distributions.normal.Normal(mu, std)
        hidden = normal_dist.rsample()
        
        # Reshaping
        result = hidden.view((task_size, -1, num_latents))
        mu = mu.view((task_size, -1, num_latents))
        std = std.view((task_size, -1, num_latents))

        # kl loss
        self.kl_loss = self.kld(log_alpha)

        return result

    def max_alpha(self):
        log_alpha = self.log_sigma2 - self.theta ** 2
        return torch.max(log_alpha.exp())