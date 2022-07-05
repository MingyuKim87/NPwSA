import numpy as np
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# for log gamma function
from scipy.special import loggamma


class Attention(nn.Module):
    '''
        The Attention module
    '''

    def __init__(self, device, embedding_type, layer_sizes, dim, att_type, scale=1., normalize=True, **kargs):
        '''
            Create attention modules

            Takes in context inputs, target inputs, and representations of each context input / output pairs (for attention)
            for obtaining outputs aggregating representation of the context data

            Args:
                embedding_type : embedding using MLP or not. (Embedding = 'MLP', not embedding = 'Identity')
                dim : a dict type have the last dimension size of "query" and "value"
                layer_sizes : if 'MLP', layer sizes
                att_type : type of attention, one of the following ['uniform', 'laplace', 'dot_product', 'multihead']
                scale : float, scale of attention
                normalize : boolean, determining whether to : 
                    1. apply "Softmax" for weights in attention.
                    2. apply the predefined operation forcing all weights in [0,1]
                num_heads : integer, if 'multihead', number of heads.
                hid_size : "context_prior" hidden size = 10
        '''

        super(Attention, self).__init__()
        
        #device
        self.device = device
        
        self._embedding_type = embedding_type
        self._layer_sizes = layer_sizes

        if self._embedding_type == 'mlp':
            # MLP embedding architecture
            self._layers = nn.ModuleList([])
            for i in range(len(self._layer_sizes)):
                if i == 0:
                    self._layers.append(nn.Linear(dim['q_last_dim'], layer_sizes[0]))
                else:
                    self._layers.append(nn.Linear(layer_sizes[i-1], layer_sizes[i]))

        self._att_type = att_type
        self._scale = scale
        self._normalize = normalize

        # Multi-head Attention
        if self._att_type == 'multihead' or \
            self._att_type == 'soft_multihead_log_normal' or \
            self._att_type == 'soft_multihead_weibull':
            
            # Define number of heads
            self._num_heads = kargs.get("num_heads", 8)

            # head_size
            self.head_size = dim['v_last_dim'] / self._num_heads

            # Initialize wieghts
            self.weights_q = nn.Linear(self._layer_sizes[-1], self._num_heads * self._layer_sizes[-1])
            self.weights_k = nn.Linear(self._layer_sizes[-1], self._num_heads * self._layer_sizes[-1])
            self.weights_v = nn.Linear(dim['v_last_dim'], self._num_heads * dim['v_last_dim'])

        # Bayesian Attnetion Modules
        if self._att_type.find("log_normal") >= 0 or \
            self._att_type.find("weibull") >= 0 :
            
            # prior type : contextual or else
            self.prior_type = kargs.get("prior_type", "contextual")
            
            # hyper-parameters
            self.eps = kargs.get("eps", 1e-20)
            self.training = kargs.get("training", 1.0)

            if self._att_type.find("log_normal") >= 0:
                self.sigma_normal_posterior = \
                    kargs.get("simga_normal_posterior", 1.0)
                self.sigma_normal_prior = \
                    kargs.get("simga_normal_prior", 1.0)
            else:
                self.k_weibull = kargs.get("k_weibull", 30)

            # parameters
            self.hid_size = kargs.get("hid_size", 10)
            self.initializer = torch.nn.init.xavier_normal_
            self.weights_k_1 = nn.Linear(self._layer_sizes[-1], self.hid_size)
            self.weights_k_2 = nn.Linear(self.hid_size, 1)

            # Initialization
            self.initializer(self.weights_k_1.weight)
            self.initializer(self.weights_k_2.weight)

            self.coef_drop = kargs.get("coef_drop", 0.0)
            self.coef_dropout = torch.nn.Dropout(1-self.coef_drop) if self.coef_drop != 0.0 else None
            self.in_drop = kargs.get("in_drop", 0.0) 
            self.in_dropout = torch.nn.Dropout(1-self.in_drop) if self.in_drop != 0.0 else None
            self.kl_loss = None

    
    def forward(self, x1, x2, v):
        '''
        Args:
            q : queries, tensor of shape # [Batch_size, # of total_points, x_size(dim)]
            k : keys, tensor of shape # [Batch_size, # of context_points, x_size(dim)]
            v : values, tensor of shape # [Batch_size, # of context_points, representation dim(latent_dim)]

        Returns:
            tensor of shape # [batch_size, # of total_points, x_size(dim)]

        Raise:
            NameError : The argument for rep / type was invalid.
        '''

        if self._embedding_type == 'identity':
            q, k = (x1, x2)
        elif self._embedding_type == 'mlp':
            q, k = (x1, x2)

            # Set dimension
            # batch, _, filter_size = q.size()
            
            # q = q.view((-1, filter_size))
            # k = k.view((-1, filter_size))

            # Set dimension (New)
            input_shape = q.size()
            
            q = q.view((-1, input_shape[-1]))
            k = k.view((-1, input_shape[-1]))

            # embedding using MLP
            for i, layer in enumerate(self._layers):
                if i == len(self._layer_sizes)-1:
                    q = layer(q)
                    k = layer(k)
                else:
                    q = F.relu(layer(q))
                    k = F.relu(layer(k))

            # Restoring dimension
            # q = q.view((batch, -1, self._layer_sizes[-1]))
            # k = k.view((batch, -1, self._layer_sizes[-1]))

            # Restoring dimension (New)
            q = q.view((*input_shape[:-2], -1, self._layer_sizes[-1]))
            k = k.view((*input_shape[:-2], -1, self._layer_sizes[-1]))

        else:
            raise NameError("'rep' not including ['identity', 'mlp']")

        if self._att_type == 'uniform':
            representation, weights = self.uniform_attention(q, v)
        elif self._att_type == 'laplace':
            representation, weights = self.laplace_attention(q, k, v, self._scale, self._normalize)
        elif self._att_type == 'dot_prodcut':
            representation, weights = self.dot_product_attention(q, k, v, self._normalize)
        elif self._att_type == 'multihead':
            representation, weights = self.multihead_attention(q, k, v)
        elif self._att_type == 'soft_attention_log_normal':
            representation, weights = self.soft_attention_log_normal(q, k, v, self._scale, self._normalize, \
                self.prior_type, self.eps, self.training, \
                self.sigma_normal_posterior, self.sigma_normal_prior)
        elif self._att_type == 'soft_attention_weibull':
            representation, weights = self.soft_attention_weibull(q, k, v, self._scale, self._normalize, \
                self.prior_type, self.eps, self.training,\
                self.k_weibull)
        elif self._att_type == 'soft_multihead_log_normal':
            representation, weights = self.soft_multihead_attention_log_normal(q, k, v, self._scale, self._normalize, \
                self.prior_type, self.eps, self.training, \
                self.sigma_normal_posterior, self.sigma_normal_prior)
        elif self._att_type == 'soft_multihead_weibull':
            representation, weights = self.soft_multihead_attention_weibull(q, k, v, self._scale, self._normalize, \
                self.prior_type, self.eps, self.training,\
                self.k_weibull)

        else:
            raise NameError(("'att_type', not including ['uniform', 'laplace', 'dot_product', 'multihead', 'soft_attention']"))

        return representation, weights

    def uniform_attention(self, q, v):
        '''
        Uniform attention. Equivalent to Neural Processes --> Aggregation


        Args:
            q : queries, tensor of shape # [Batch_size, # of total_points, y_size(dim)]
            v : values, tensor of shape # [Batch_size, # of context_points, representation dim(latent_dim)]

        Returns : 
            tensor of shape # [Batch_size, # of total point, representation dim(latent_dim)]
        '''

        total_points = q.size()[1] # number of points (shots)

        # Aggregation operation
        #rep = torch.mean(v, dim=1, keepdims=True) 
        rep = v.mean(dim=1, keepdims=True) # rep [batch_size, 1, representation_dim(latent_dim)]

        # 뻥튀기
        rep = rep.repeat(1, total_points, 1) # rep [batch_size, total_points, representation_dim(latent_dim)]

        return rep, None

    def laplace_attention(self, q, k, v, scale, normalize):
        '''
        Computes laplace expontial attention. 

        Args:
            q : queries, tensor of shape # [Batch_size, # of total_points, x_size(dim)]
            k : keys, tensor of shape # [Batch_size, # of context_points, x_size(dim)]
            v : values, tensor of shape # [Batch_size, # of context_points, representation dim(latent_dim)]
            scale : scalar (float), L1 distance
            normalize : boolean, all weights sum to 1 or not.

        Returns:
            tensor of shape # [Batch_size, # of total_points, representation dim(latent_dim)]
        '''

        k = k.unsqueeze(1) #[batch_size, 1, # of context_points, x_size(dim)]
        q = q.unsqueeze(2) #[batch_size, # of total_points, 1, x_size(dim)]

        # [batch_size, # of total_point, # of context_point, x_size(dim)]
        # broading cast (element-wise distance) e.g) matmul(X, XT)
        unnorm_weights = -1 * torch.abs(torch.div(torch.add(k, torch.mul(q, -1)), scale)) 
        #unnorm_weights = torch.sum(unnorm_weights, dim=-1)
        unnorm_weights = unnorm_weights.sum(dim=-1)

        if normalize:
            weight_fn = F.softmax
        else:
            weight_fn = F.tanh

        weights = weight_fn(unnorm_weights)
        rep = torch.matmul(weights, v)

        return rep, weights


    def dot_product_attention(self, q, k, v, normalize):
        '''
        Computes laplace expontial attention. 

        Args:
            q : queries, tensor of shape # [Batch_size, # of total_points, x_size(dim)]
            k : keys, tensor of shape # [Batch_size, # of context_points, x_size(dim)]
            v : values, tensor of shape # [Batch_size, # of context_points, representation dim(latent_dim)]
            scale : scalar (float), L1 distance
            normalize : boolean, all weights sum to 1 or not.

            if multihead uses :
                q : queries, tensor of shape #[task_size, num_heads, # of total point, q_latent_dim]
                k : keys, tensor of shape #[task_size, num_heads, # of context point, q_latent_dim]
                v : values, tensor of shape #[task_size, num_heads, # of context point, q_latent_dim]

        Returns:
            tensor of shape # [Batch_size, # of total_points, representation dim(latent_dim)]
        '''

        # Scale parameterd_k
        d_k = q.size()[-1]
        scale = math.sqrt(d_k)

        # Transpose 
            # [batch_size, x_size(dim), # of context_points]
            # if multihead, #[task_size, num_heads, q_latent_dim , # of context point]
        k = torch.transpose(k, -1, -2) 

            # if multihead, #[task_size, num_heads, # of total point, # of context point]
        unnorm_weights = torch.div(torch.matmul(q, k), scale) 

        if normalize:
            weight_fn = F.softmax
        else:
            weight_fn = F.tanh

        # if multihead, [batch_size, num_heads, # of total_points, # of context_points]
        weights = weight_fn(unnorm_weights, dim=-1) 

        # if multihead, [batch_size, num_heads, # of total_points, q_latent_dim]
        rep = torch.matmul(weights, v)

        return rep, weights

    def multihead_attention(self, q, k, v):
        '''
        Computes multi-head attention. Implemented by Tensorflow Tutorials.

        Args:
            q : queries, tensor of shape # [Batch_size, # of total_points, x_size(dim)]
            k : keys, tensor of shape # [Batch_size, # of context_points, x_size(dim)]
            v : values, tensor of shape # [Batch_size, # of context_points, representation dim(latent_dim)]
            num_heads : the number of heads, int, it should divide representation_dim(dim)            

        Returns:
            tensor of shape # [Batch_size, # of total_points, representation dim(latent_dim)]
         '''

        # dim
        ndim = q.dim()
        
        # Size
        q_dim = q.size()
        k_dim = k.size()
        v_dim = v.size()

        
        # Parameters
        self.head_size

        # Initialization of results
        representation = 1.0

        # forward
        '''
            q_dim, k_dim, v_dim 정보들을 모두 수정하기 (정보가 없음)
        '''
        if ndim==3: # (task_size, # of points, x_dim)
            q_prime = torch.transpose(self.weights_q(q).view(q_dim[0], -1, self._num_heads, q_dim[-1]), 1, 2) #[task_size, num_heads, # of total point, q_latent_dim]
            k_prime = torch.transpose(self.weights_k(k).view(k_dim[0], -1, self._num_heads, k_dim[-1]), 1, 2) #[task_size, num_heads, # of context point, q_latent_dim]
            v_prime = torch.transpose(self.weights_v(v).view(v_dim[0], -1, self._num_heads, v_dim[-1]), 1, 2) #[task_size, num_heads, # of context point, q_latent_dim]
        else: # for importance weighted : (n_iter, task_size, # of points, x_dim)
            q_prime = torch.transpose(self.weights_q(q).view(*q_dim[:-2], -1, self._num_heads, q_dim[-1]), 2, 3) #[task_size, num_heads, # of total point, q_latent_dim]
            k_prime = torch.transpose(self.weights_k(k).view(*k_dim[:-2], -1, self._num_heads, k_dim[-1]), 2, 3) #[task_size, num_heads, # of context point, q_latent_dim]
            v_prime = torch.transpose(self.weights_v(v).view(*v_dim[:-2], -1, self._num_heads, v_dim[-1]), 2, 3) #[task_size, num_heads, # of context point, q_latent_dim]

        '''
        q_prime = torch.transpose(torch.reshape(torch.matmul(q, self.weights_q), self.q_dim[0], -1, num_heads, self.q_dim[-1]), 1, 2)
        k_prime = torch.transpose(torch.reshape(torch.matmul(k, self.weights_k), self.k_dim[0], -1, num_heads, self.k_dim[-1]), 1, 2)
        v_prime = torch.transpose(torch.reshape(torch.matmul(v, self.weights_v), self.v_dim[0], -1, num_heads, self.v_dim[-1]), 1, 2)
        '''

        output, weights = self.dot_product_attention(q_prime, k_prime, v_prime, normalize=True)
        
        # mean
        # output = output.mean(dim=1)
        # weights = weights.mean(dim=1)
        
        # mean
        output = output.mean(dim=-3)
        weights = weights.mean(dim=-3)

        representation = output + representation 


        #[batch_size, # of total_point, representation dim(latent_dim)] <= [batch_size, # of total_point, # of context_point, x_size(dim)] * [Batch_size, # of context_points, representation dim(latent_dim)]
        return representation, weights


    def soft_attention_log_normal(self, q, k, v, scale, normalize,\
        prior_type="contextual", eps=1e-20, training=1.0, 
        sigma_normal_posterior=1.0, sigma_normal_prior=1.0):
        '''
            Bayesian attention (reparameterization trick on log_normal)

        Args:
            q : queries, tensor of shape # [Batch_size, # of total_points, x_size(dim)]
            k : keys, tensor of shape # [Batch_size, # of context_points, x_size(dim)]
            v : values, tensor of shape # [Batch_size, # of context_points, representation dim(latent_dim)]
            scale : scalar (float), L1 distance
            normalize : boolean, all weights sum to 1 or not.

        Returns:
            tensor of shape # [Batch_size, # of total_points, representation dim(latent_dim)]
         '''

        # parameters
        eps = eps
        training = training

        # KL and reparameterization sampling
        sigma_normal_posterior = sigma_normal_posterior
        sigma_normal_prior = sigma_normal_prior
        
        # Weights
        key = k

        k = k.unsqueeze(1) #[batch_size, 1, # of context_points, x_size(dim)]
        q = q.unsqueeze(2) #[batch_size, # of total_points, 1, x_size(dim)]

        # [batch_size, # of total_point, # of context_point, x_size(dim)]
        # broading cast (element-wise distance) e.g) matmul(X, XT)
        unnorm_weights = -1 * torch.abs(torch.div(torch.add(k, torch.mul(q, -1)), scale)) 
        #unnorm_weights = torch.sum(unnorm_weights, dim=-1)
        unnorm_weights = unnorm_weights.sum(dim=-1)

        if normalize:
            weight_fn = F.softmax
        else:
            weight_fn = F.tanh

        weights = weight_fn(unnorm_weights) #[batch_size, # of total_point, # of context_point]

        # Posterior
        logprob = torch.log(weights + eps) #[batch_size, # of total_point, # of context_point]

        # Prior from "key"
        if prior_type == "contextual":
            dot_mu = F.relu(self.weights_k_1(key))
            dot_mu = self.weights_k_2(dot_mu)
            dot_mu = F.softmax(torch.transpose(dot_mu, 1, 2), dim=-1) # [batch_size, 1, # of total_points]
                
            mean_normal_prior = torch.log(dot_mu + eps) # [batch_size, 1, # of total_points]
            prior_weights = dot_mu

        else:
            alpha_gamma = 1.0
            mean_normal_prior = 0.0

        # Posterior sampling and KL
            # shape
        logprob_size = list(logprob.size())
            
        # Random number for rsamples
        u_lognormal = torch.randn(*logprob_size).to(self.device)
            
        # Normal
        mean_normal_posterior = logprob - (sigma_normal_posterior ** 2) / 2

        # rsamples    
        sample_normal = mean_normal_posterior \
            + training * (sigma_normal_posterior * u_lognormal) \
            + (1-training) * (sigma_normal_posterior ** 2) / 2 

        # satisfy the simplex constraint.
        weights = F.softmax(sample_normal) #[batch_size, # of total_point, # of context_point]

        # KLD
        KL = np.log(sigma_normal_prior / sigma_normal_posterior + eps) \
            + (sigma_normal_posterior ** 2 + (mean_normal_prior - mean_normal_posterior)**2) / (2 * sigma_normal_prior ** 2) \
            - 0.5

        mask = torch.where(KL > -1e7, torch.ones_like(KL), torch.zeros_like(KL))
        KL = KL * mask
        KL_backward = KL.sum() / mask.sum()

        if self.coef_drop != 0.0:
            self.coef_dropout(weights)
        
        if self.in_drop != 0.0:
            self.in_dropout(k)

        # Attentions
        rep = torch.matmul(weights, v)

        # add kl loss
        self.kl_loss = KL_backward

        return F.relu(rep), weights

    def soft_attention_weibull(self, q, k, v, scale, normalize,\
        prior_type="contextual",eps=1e-20, training=1.0,\
        k_weibull=30):
        
        '''
            Bayesian attention on reparameterization trick (weibull)

        Args:
            q : queries, tensor of shape # [Batch_size, # of total_points, x_size(dim)]
            k : keys, tensor of shape # [Batch_size, # of context_points, x_size(dim)]
            v : values, tensor of shape # [Batch_size, # of context_points, representation dim(latent_dim)]
            scale : scalar (float), L1 distance
            normalize : boolean, all weights sum to 1 or not.

        Returns:
            tensor of shape # [Batch_size, # of total_points, representation dim(latent_dim)]
         '''

        # parameters
        eps = eps
        k_weibull = k_weibull
        training = training

        key = k

        k = k.unsqueeze(1) #[batch_size, 1, # of context_points, x_size(dim)]
        q = q.unsqueeze(2) #[batch_size, # of total_points, 1, x_size(dim)]

        # [batch_size, # of total_point, # of context_point, x_size(dim)]
        # broading cast (element-wise distance) e.g) matmul(X, XT)
        unnorm_weights = -1 * torch.abs(torch.div(torch.add(k, torch.mul(q, -1)), scale)) 
        #unnorm_weights = torch.sum(unnorm_weights, dim=-1)
        unnorm_weights = unnorm_weights.sum(dim=-1)

        if normalize:
            weight_fn = F.softmax
        else:
            weight_fn = F.tanh

        weights = weight_fn(unnorm_weights) #[batch_size, # of total_point, # of context_point]

        # Posterior
        logprob = torch.log(weights + eps) #[batch_size, # of total_point, # of context_point]

        # Prior from "key"
        if prior_type == "contextual":
            dot_gamma = F.relu(self.weights_k_1(key)) # [batch_size, # of total_point, hidden]
            dot_gamma = self.weights_k_2(dot_gamma) # [batch_size, # of total_point, 1]
            dot_gamma = torch.transpose(dot_gamma, 1, 2) # [batch_size, 1, # of total_point]

            alpha_gamma = F.softmax(dot_gamma, dim=-1) # [batch_size, 1, # of total_points]
            prior_weights = alpha_gamma / alpha_gamma.sum(dim=-1, keepdims=True) 

        else:
            alpha_gamma = 1.0
            mean_normal_prior = 0.0

        # Posterior sampling and KL
            # posterior shape
        logprob_size = list(logprob.size())
            
        # Random number for rsamples
        u_weibull = torch.rand(*logprob_size).to(self.device)
        lambda_weibull = torch.exp(logprob) / np.exp(loggamma(1 + (1.0/k_weibull)))
            
        # rsamples
        sample_weibull = lambda_weibull * (
            + training * torch.exp(1.0 \
                / k_weibull * torch.log(-torch.log(1.0 - u_weibull + eps) + eps)) \
            + (1-training) * np.exp(loggamma(1.0 + 1.0/k_weibull)))

        # satisfy the simplex constraint    
        weights = sample_weibull / sample_weibull.sum(dim=-1, keepdim=True) #[batch_size, # of total_point, # of context_point]

        KL = - (alpha_gamma * torch.log(lambda_weibull + eps) \
            - np.euler_gamma * alpha_gamma / k_weibull \
            - np.log(k_weibull + eps) \
            - 1.0 * lambda_weibull * np.exp(loggamma(1 + 1.0/k_weibull)) \
            + np.euler_gamma + 1.0 \
            + alpha_gamma * np.log(1.0 + eps) - torch.lgamma(alpha_gamma + eps))

        mask = torch.where(KL > -1e7, torch.ones_like(KL), torch.zeros_like(KL))
        KL = KL * mask
        KL_backward = KL.sum() / mask.sum()

        if self.coef_drop != 0.0:
            self.coef_dropout(weights)
        
        if self.in_drop != 0.0:
            self.in_dropout(k)

        rep = torch.matmul(weights, v)

        # add kl loss
        self.kl_loss = KL_backward

        return F.relu(rep), weights


    def soft_attention_log_normal_2(self, q, k, v, scale, normalize,\
        prior_type="contextual", eps=1e-20, training=1.0,\
        sigma_normal_posterior=1.0, sigma_normal_prior=1.0):

        # Weights
        key = k
        
        # Scale parameterd_k
        d_k = q.size()[-1]
        scale = math.sqrt(d_k)

        # Transpose 
            # change the q_latent_dim axis into # of context_points  
            # [batch_size, x_size(dim), # of context_points]
            # if multihead, #[task_size, num_heads, q_latent_dim , # of context point]
        k = torch.transpose(k, -1, -2) 

            # if multihead, #[task_size, num_heads, # of total point, # of context point]
        unnorm_weights = torch.div(torch.matmul(q, k), scale)

        if normalize:
            weight_fn = F.softmax
        else:
            weight_fn = F.tanh

        weights = weight_fn(unnorm_weights)  

        # Posterior
        logprob = torch.log(weights + eps) #[batch_size, # of total_point, # of context_point]

        # Prior from "key"
        if prior_type == "contextual":
            dot_mu = F.relu(self.weights_k_1(torch.transpose(k, -1, -2)))
            dot_mu = self.weights_k_2(dot_mu)
            dot_mu = F.softmax(torch.transpose(dot_mu, -1, -2), dim=-1) # [batch_size, 1, # of total_points]
                
            mean_normal_prior = torch.log(dot_mu + eps) # [batch_size, 1, # of total_points]
            prior_weights = dot_mu

        else:
            alpha_gamma = 1.0
            mean_normal_prior = 0.0

        # Posterior sampling and KL
            # shape
        logprob_size = list(logprob.size())
            
        # Random number for rsamples
        u_lognormal = torch.randn(*logprob_size).to(self.device)
            
        # Normal
        mean_normal_posterior = logprob - (sigma_normal_posterior ** 2) / 2

        # rsamples    
        sample_normal = mean_normal_posterior \
            + training * (sigma_normal_posterior * u_lognormal) \
            + (1-training) * (sigma_normal_posterior ** 2) / 2 

        # satisfy the simplex constraint.
        weights = F.softmax(sample_normal) #[batch_size, # of total_point, # of context_point]

        # KLD
        KL = np.log(sigma_normal_prior / sigma_normal_posterior + eps) \
            + (sigma_normal_posterior ** 2 + (mean_normal_prior - mean_normal_posterior)**2) / (2 * sigma_normal_prior ** 2) \
            - 0.5

        mask = torch.where(KL > -1e7, torch.ones_like(KL), torch.zeros_like(KL))
        KL = KL * mask
        KL_backward = KL.sum() / mask.sum()

        if self.coef_drop != 0.0:
            self.coef_dropout(weights)
        
        if self.in_drop != 0.0:
            self.in_dropout(k)

        # Attentions
        rep = torch.matmul(weights, v)

        # add kl loss
        self.kl_loss = KL_backward

        return rep, weights

    def soft_attention_weibull_2(self, q, k, v, scale, normalize,\
        prior_type="contextual",eps=1e-20, training=1.0,\
        k_weibull=30):

        # Weights
        key = k
        
        # Scale parameterd_k
        d_k = q.size()[-1]
        scale = math.sqrt(d_k)

        # Transpose 
            # [batch_size, x_size(dim), # of context_points]
            # if multihead, #[task_size, num_heads, q_latent_dim , # of context point]
        k = torch.transpose(k, -1, -2) 

            # if multihead, #[task_size, num_heads, # of total point, # of context point]
        unnorm_weights = torch.div(torch.matmul(q, k), scale)

        if normalize:
            weight_fn = F.softmax
        else:
            weight_fn = F.tanh

        weights = weight_fn(unnorm_weights)  

        # Posterior
        logprob = torch.log(weights + eps) #[batch_size, # of total_point, # of context_point]

        # Prior from "key"
        if prior_type == "contextual":
            dot_gamma = F.relu(self.weights_k_1(torch.transpose(k,-1,-2))) # [batch_size, # of total_point, hidden]
            dot_gamma = self.weights_k_2(dot_gamma) # [batch_size, # of total_point, 1]
            dot_gamma = torch.transpose(dot_gamma, -1, -2) # [batch_size, 1, # of total_point]

            alpha_gamma = F.softmax(dot_gamma, dim=-1) # [batch_size, 1, # of total_points]
            prior_weights = alpha_gamma / alpha_gamma.sum(dim=-1, keepdims=True) 

        else:
            alpha_gamma = 1.0
            mean_normal_prior = 0.0

        # Posterior sampling and KL
            # posterior shape
        logprob_size = list(logprob.size())
            
        # Random number for rsamples
        u_weibull = torch.rand(*logprob_size).to(self.device)
        lambda_weibull = torch.exp(logprob) / np.exp(loggamma(1 + (1.0/k_weibull)))
            
        # rsamples
        sample_weibull = lambda_weibull * (
            + training * torch.exp(1.0 \
                / k_weibull * torch.log(-torch.log(1.0 - u_weibull + eps) + eps)) \
            + (1-training) * np.exp(loggamma(1.0 + 1.0/k_weibull)))

        # satisfy the simplex constraint    
        weights = sample_weibull / sample_weibull.sum(dim=-1, keepdim=True) #[batch_size, # of total_point, # of context_point]

        KL = - (alpha_gamma * torch.log(lambda_weibull + eps) \
            - np.euler_gamma * alpha_gamma / k_weibull \
            - np.log(k_weibull + eps) \
            - 1.0 * lambda_weibull * np.exp(loggamma(1 + 1.0/k_weibull)) \
            + np.euler_gamma + 1.0 \
            + alpha_gamma * np.log(1.0 + eps) - torch.lgamma(alpha_gamma + eps))

        mask = torch.where(KL > -1e7, torch.ones_like(KL), torch.zeros_like(KL))
        KL = KL * mask
        KL_backward = KL.sum() / mask.sum()

        if self.coef_drop != 0.0:
            self.coef_dropout(weights)
        
        if self.in_drop != 0.0:
            self.in_dropout(k)

        rep = torch.matmul(weights, v)

        # add kl loss
        self.kl_loss = KL_backward

        return rep, weights

    def soft_multihead_attention_log_normal(self, q, k, v, scale, normalize,\
        prior_type="contextual", eps=1e-20, training=1.0, 
        sigma_normal_posterior=1.0, sigma_normal_prior=1.0):

        '''
        Computes multi-head attention. Implemented by Tensorflow Tutorials.
        Based on bayesian dot-product attentions

        Args:
            q : queries, tensor of shape # [Batch_size, # of total_points, x_size(dim)]
            k : keys, tensor of shape # [Batch_size, # of context_points, x_size(dim)]
            v : values, tensor of shape # [Batch_size, # of context_points, representation dim(latent_dim)]
            num_heads : the number of heads, int, it should divide representation_dim(dim)            

        Returns:
            tensor of shape # [Batch_size, # of total_points, representation dim(latent_dim)]
         '''

        '''
            q_dim, k_dim, v_dim 정보들을 모두 수정하기 (정보가 없음)
        '''
        
        # dim
        ndim = q.dim()
        
        # Size
        q_dim = q.size()
        k_dim = k.size()
        v_dim = v.size()

        
        # Parameters
        self.head_size

        # Initialization of results
        representation = 1.0
        
        # forward
        '''
            q_dim, k_dim, v_dim 정보들을 모두 수정하기 (정보가 없음)
        '''
        if ndim == 3: # (task_size, # of points, x_dim)
            q_prime = torch.transpose(self.weights_q(q).view(q_dim[0], -1, self._num_heads, q_dim[-1]), 1, 2) #[task_size, num_heads, # of total point, q_latent_dim]
            k_prime = torch.transpose(self.weights_k(k).view(k_dim[0], -1, self._num_heads, k_dim[-1]), 1, 2) #[task_size, num_heads, # of context point, q_latent_dim]
            v_prime = torch.transpose(self.weights_v(v).view(v_dim[0], -1, self._num_heads, v_dim[-1]), 1, 2) #[task_size, num_heads, # of context point, q_latent_dim]
        else: # for importance weighted : (n_iter, task_size, # of points, x_dim)
            q_prime = torch.transpose(self.weights_q(q).view(*q_dim[:-2], -1, self._num_heads, q_dim[-1]), -3, -2) #[task_size, num_heads, # of total point, q_latent_dim]
            k_prime = torch.transpose(self.weights_k(k).view(*k_dim[:-2], -1, self._num_heads, k_dim[-1]), -3, -2) #[task_size, num_heads, # of context point, q_latent_dim]
            v_prime = torch.transpose(self.weights_v(v).view(*v_dim[:-2], -1, self._num_heads, v_dim[-1]), -3, -2) #[task_size, num_heads, # of context point, q_latent_dim]

        # soft_attention_log_normal
        output, weights = self.soft_attention_log_normal_2(q_prime, k_prime, v_prime, scale=scale, normalize=normalize,\
            prior_type=prior_type, eps=eps, training=training, \
            sigma_normal_posterior=sigma_normal_posterior,\
            sigma_normal_prior=sigma_normal_prior)

        # mean
        output = output.mean(dim=-3)
        weights = weights.mean(dim=-3)

        representation = output + representation 
        
        #[batch_size, # of total_point, representation dim(latent_dim)] <= [batch_size, # of total_point, # of context_point, x_size(dim)] * [Batch_size, # of context_points, representation dim(latent_dim)]
        return representation, weights


    def soft_multihead_attention_weibull(self, q, k, v, scale, normalize,\
        prior_type="contextual",eps=1e-20, training=1.0,\
        k_weibull=30):

        # dim
        ndim = q.dim()
        
        # Size
        q_dim = q.size()
        k_dim = k.size()
        v_dim = v.size()

        # Parameters
        self.head_size

        # Initialization of results
        representation = 1.0
        
        # forward
        '''
            q_dim, k_dim, v_dim 정보들을 모두 수정하기 (정보가 없음)
        '''
        if ndim == 3:
            q_prime = torch.transpose(self.weights_q(q).view(q_dim[0], -1, self._num_heads, q_dim[-1]), 1, 2) #[task_size, num_heads, # of total point, q_latent_dim]
            k_prime = torch.transpose(self.weights_k(k).view(k_dim[0], -1, self._num_heads, k_dim[-1]), 1, 2) #[task_size, num_heads, # of context point, q_latent_dim]
            v_prime = torch.transpose(self.weights_v(v).view(v_dim[0], -1, self._num_heads, v_dim[-1]), 1, 2) #[task_size, num_heads, # of context point, q_latent_dim]
        else:
            q_prime = torch.transpose(self.weights_q(q).view(*q_dim[:-2], -1, self._num_heads, q_dim[-1]), -3, -2) #[task_size, num_heads, # of total point, q_latent_dim]
            k_prime = torch.transpose(self.weights_k(k).view(*k_dim[:-2], -1, self._num_heads, k_dim[-1]), -3, -2) #[task_size, num_heads, # of context point, q_latent_dim]
            v_prime = torch.transpose(self.weights_v(v).view(*v_dim[:-2], -1, self._num_heads, v_dim[-1]), -3, -2) #[task_size, num_heads, # of context point, q_latent_dim]

        # soft_attention_log_normal
        output, weights = self.soft_attention_weibull_2(q_prime, k_prime, v_prime, scale=scale, normalize=normalize,\
            prior_type=prior_type, eps=eps, training=training, \
            k_weibull=k_weibull)

        # mean
        output = output.mean(dim=-3)
        weights = weights.mean(dim=-3)

        representation = output + representation 
        
        #[batch_size, # of total_point, representation dim(latent_dim)] <= [batch_size, # of total_point, # of context_point, x_size(dim)] * [Batch_size, # of context_points, representation dim(latent_dim)]
        return representation, weights






    
            













                











        

        

        
        
        
        
         






if __name__ == '__main__':
    a = np.ones((3,2,2))
    b = np.random.normal(size=(2,2))

    c = np.matmul(a,b)

    print(c.shape)
    print(c)