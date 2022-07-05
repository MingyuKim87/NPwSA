import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from torch.optim import Optimizer

from .parts.abstract_np import abstract_NPs
from .parts.modules import Stochastic_Encoder, Decoder
from .parts.criterion import elbo_loss

class NP(abstract_NPs):
    '''
        NP model
    '''

    def __init__(self, encoder_input_dim, encoder_layer_sizes, num_latents, \
        decoder_input_dim, decoder_layer_sizes, output_dim, device):
        '''
            Args:
                [encoder part]
                encoder_input_dim : x_dim + y_dim
                stochastic_encoder_layer_sizes : layer sizes
                num_latents : num_latents
                [decoder part]
                decoder_layer_size : layer sizes
                decoder_input_dim : x_dim + num_latents
            
            Return : 
                model
        '''

        super().__init__()
        self._stochastic_encoder = Stochastic_Encoder(encoder_input_dim, encoder_layer_sizes, num_latents)
        self._decoder = Decoder(decoder_input_dim, decoder_layer_sizes, output_dim)
        
        self.device = device
        
    def forward(self, context_x, context_y, target_x, num_total_points, target_y = None):
        '''
            Returns mu and sigma from the predictive distribution

            Args:
                query : (context_x, context_y), target_x)
                    context_x : [batch_size, num_context, x_size(dim)] at 1D regression
                    context_y : [batch_size, num_context, y_size(dim)] at 1D regression
                    target_x : [batch_size, num_total_point, x_size(dim)] at 1D regression
                num_total_points : the number of target points
                target_y : [batch_size, num_total_points, y_size(dim)]

            Returns:
                log_p : multivariate gaussian dist. Sample from this distribution has [batch_size, the number of points]
                mu : mean of multivariate distribution [batch_size, the number of points, y_size(dim)]
                sigma : std of multivariate didstribution [batch_size, the number of points, y_size(dim)]
        '''
        # tensor dim
        ndim = context_x.dim()

        if target_y is not None:
            # training model
            self.train(True)

            # Pass query through stochastic encoders and prior dist. 
            prior = self._stochastic_encoder(context_x, context_y) #[task_size, latent_dim]  
            posterior = self._stochastic_encoder(target_x, target_y) #[task_size, latent_dim]  

            # MC Sampling
            stochastic_rep = prior.rsample() #[task_size, num_latents]

            # 뻥튀기
            if ndim == 3:
                stochastic_rep = torch.unsqueeze(stochastic_rep, dim=-2).repeat(1, num_total_points, 1)  #[batch_size, the number of points, latent_dim]
            else:
                stochastic_rep = torch.unsqueeze(stochastic_rep, dim=-2).repeat(1, 1, num_total_points, 1)  #[n_samples, batch_size, the number of points, latent_dim]
            
            # Decoding
            mu, sigma = self._decoder(target_x, stochastic_rep)

            # Distribution
            p_y_pred = torch.distributions.Normal(mu, sigma)

            
            return p_y_pred, posterior, prior, None, stochastic_rep

        else:
            # evaluation mode
            self.train(False)
            
            prior = self._stochastic_encoder(context_x, context_y) 

            #MC Sampling
            stochastic_rep = prior.rsample() #[task_size, num_latents]
            
            # 뻥튀기
            stochastic_rep = torch.unsqueeze(stochastic_rep, dim=1).repeat(1, num_total_points, 1)  #[batch_size, the number of points, latent_dim]
            
            # Decoding
            mu, sigma = self._decoder(target_x, stochastic_rep)

            # Distribution
            p_y_pred = torch.distributions.Normal(mu, sigma)

            return p_y_pred, mu, sigma, None, stochastic_rep