import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable


class Deterministic_Encoder(nn.Module):
    '''
        Deterministic Encoder
    '''

    def __init__(self, input_dim, layer_sizes, num_latents, attention = None):
        '''
            Np deterministic encoder
            
            Args:
                input_dim : x_dim + y_dim
                layer_sizes : the array of each lyaer size in encoding MLP
                num_latents : the latent dimensionality
        '''

        super().__init__()

        # Attention
        self._attention = attention
            
        # Layer size
        self._layer_sizes = layer_sizes

        # Output dim
        self._num_latents = num_latents

        # MLP embedding architectures
        self._layers = nn.ModuleList([])
        for i in range(len(self._layer_sizes)):
            if i == 0:
                self._layers.append(nn.Linear(input_dim, self._layer_sizes[0]))
            else:
                self._layers.append(nn.Linear(self._layer_sizes[i-1], self._layer_sizes[i]))

        self._last_layer = nn.Linear(self._layer_sizes[-1], self._num_latents)

    def forward(self, context_x, context_y, target_x=None):
        '''
            Ecoding the input into representation using latent encoder

            Args:
                context_x: [batch_size, the number of observation, x_size(dimension)] 
                context_y : [batch_size, the number of observation, y_size(dimension)] 

            Returns:
                representation : [batch_size, the nmber of observation, num_lantents]
        '''

        # Concatenate x and y along the filter axises
            # Technique
        encoder_input = torch.cat((context_x, context_y), dim=-1) # [batch_size, the number of points, x_size + y_size]

        # Shape
        task_size, _, filter_size = tuple(encoder_input.shape)

        # Input
        hidden = encoder_input.view((-1, filter_size))

        # MLP embedidng for NP
        for i, layer in enumerate(self._layers):
            hidden = F.relu(layer(hidden))

        # Last layer
        hidden = self._last_layer(hidden)

        # Reshaping
        hidden = hidden.view((task_size, -1, self._num_latents)) # [batch_size, the number of point, the last element in the list]

        if self._attention is not None:
            # Attentive neural process
            rep, weights = self._attention(target_x, context_x, hidden)
            
        else:
            # Neural Processes
                # Aggregation of representation
            rep = hidden.mean(dim=1)
            weights = None

        return rep, weights

class Stochastic_Encoder(nn.Module):
    '''
        Latent Encoder
    '''

    def __init__(self, input_dim, layer_sizes, num_latents):
        '''
            Np Encoder

            Args:
                input_dim : x_dim + y_dim
                layer_sizes : the array of each layer size in encoding MLP
                num_latents : the latent dimensionality
        '''

        super(Stochastic_Encoder, self).__init__()

        # Layer size
        self._layer_sizes = layer_sizes

        # Output dim
        self._num_latents = num_latents

        # MLP embedding architectures
        self._layers = nn.ModuleList([])
        for i in range(len(self._layer_sizes)):
            if i == 0:
                self._layers.append(nn.Linear(input_dim, self._layer_sizes[0]))
            else:
                self._layers.append(nn.Linear(self._layer_sizes[i-1], self._layer_sizes[i]))
                
        # Distribution
        self.hidden_to_mu = nn.Linear(self._layer_sizes[-1], num_latents)
        self.hidden_to_sigma = nn.Linear(self._layer_sizes[-1], num_latents)


    def forward(self, context_x, context_y):
        '''
            Ecoding the input into representation using latent encoder

            Args:
                context_x: [batch_size, the number of observation, x_size(dimension)] 
                context_y : [batch_size, the number of observation, y_size(dimension)] 

            Returns:
                A normal distribution object based on mu 
        '''

        # Concatenate x and y along the filter axises
            # Technique
        encoder_input = torch.cat((context_x, context_y), dim=-1) # [batch_size, the number of points, x_size + y_size]

        # Shape
        task_size, _, filter_size = tuple(encoder_input.shape)

        # Reshaping Inputs
        hidden = encoder_input.view((-1, filter_size))

        # MLP embedidng for NP
        for i, layer in enumerate(self._layers):
            hidden = F.relu(layer(hidden))

        # Reshaping hidden
        hidden = hidden.view(task_size, -1, self._layer_sizes[-1])

        # Aggregation
        hidden = hidden.mean(dim=1)

        # last layer
        mu = self.hidden_to_mu(hidden) # [batch_size, num_lantets]
        sigma = 0.1 + 0.9 * torch.sigmoid(self.hidden_to_sigma(hidden)) # [batch_size, num_lantets]

        # Distribution
        dist = torch.distributions.Normal(mu, sigma) 

        return dist

class Decoder(nn.Module):
    '''
        (A)NP decoder
    '''

    def __init__(self, input_dim, layer_sizes, output_dim):
        '''
            Args:
                input_dim : x_dim + latent_dim
                layer_sizes : the array of each layer size in encoding NP
        '''

        super(Decoder, self).__init__()
        self.input_dim = input_dim
        self._layer_sizes = layer_sizes
        self._output_dim = output_dim

        # MLP embedding architecture
        self._layers = nn.ModuleList([])
        for i in range(len(self._layer_sizes)):
            if i == 0:
                self._layers.append(nn.Linear(input_dim, self._layer_sizes[0]))
            else:
                self._layers.append(nn.Linear(self._layer_sizes[i-1], self._layer_sizes[i]))

        # Distribution
        self.hidden_to_mu = nn.Linear(self._layer_sizes[-1], self._output_dim)
        self.hidden_to_sigma = nn.Linear(self._layer_sizes[-1], self._output_dim)

        

    def forward(self, target_x, stochastic_rep, deterministic_rep=None):
        '''
            Decoders the individual targets

            Args:
                representation : [batch_size, the number of points, num_latents (or None)]
                target_x : [batch_size, the number of points, x_size(dim)]

            Returns:
                dist : multivariate gaussian dist. Sample from this distribution has [batch_size, the number of points, y_size(dim)]
                mu : mean of multivariate distribution [batch_size, the number of points, y_size(dim)]
                sigma : std of multivariate didstribution [batch_size, the number of points, y_size(dim)]
        '''
        # Concatenate target_x and representation        
        
        if deterministic_rep is not None:
            hidden = torch.cat((stochastic_rep, deterministic_rep, target_x), dim = -1)
        else:
            hidden = torch.cat((stochastic_rep, target_x), dim = -1)
        
        # Shape
        batch_size, _, filter_size = tuple(hidden.shape)

        # Input
        hidden = hidden.view(-1, filter_size)

        # Exceptional Treatement
        assert filter_size == self.input_dim, "You must match the dimension of input_dim and representations + target_x"
        
        # MLP embedding for NP
        for i, layer in enumerate(self._layers):
            hidden = F.relu(layer(hidden))
    
        # Forwarding mean and std
        mu = self.hidden_to_mu(hidden) # [batch_size*num_data_points, num_lantets]
        log_sigma = self.hidden_to_sigma(hidden) # [batch_size*num_data_points, num_lantets]

        # Reshaping mu and log_sigma
        mu = mu.view(batch_size, -1, self._output_dim)
        sigma = 0.1 + 0.9 * F.softplus(log_sigma.view(batch_size, -1, self._output_dim))

        return mu, sigma