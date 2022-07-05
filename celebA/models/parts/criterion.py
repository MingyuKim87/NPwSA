import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class elbo_loss(nn.Module):
    def __init__(self, model_name=None):
        super(elbo_loss, self).__init__()
        # model name
        self.model_name = None

        # loglikelihood
        self.log_likelihood = loglikelihood()

        # kld np
        self.kld = kld()

        # iwae loss
        self.iwae_loss = iwae_loss()

    def forward(self, p_y_dist, target_y, posterior_z_dist=None, prior_z_dist=None, **kargs):
        # kld_additional
        kld_additional = kargs.get("kld_additional", None)

        # Importance weighted loss
        is_iwae = kargs.get("is_iwae", False)

        if is_iwae:
            loss, mean_log_likelihood, mean_kld = self.iwae_loss(p_y_dist, target_y, posterior_z_dist, prior_z_dist)
            return loss, mean_log_likelihood, mean_kld, None
        
        if kld_additional is None:
            if posterior_z_dist == None and prior_z_dist==None:
                log_likelihood = self.log_likelihood(p_y_dist, target_y)
                loss = -1 * log_likelihood
                return loss, log_likelihood, None, None
            else:
                log_likelihood = self.log_likelihood(p_y_dist, target_y)
                kld = self.kld(posterior_z_dist, prior_z_dist)
                loss = -1 * log_likelihood + kld
                return loss, log_likelihood, kld, None
        else:
            if posterior_z_dist == None and prior_z_dist==None:
                log_likelihood = self.log_likelihood(p_y_dist, target_y)
                loss = -1 * log_likelihood + kld_additional
                return loss, log_likelihood, None, kld_additional
            else:
                log_likelihood = self.log_likelihood(p_y_dist, target_y)
                kld = self.kld(posterior_z_dist, prior_z_dist)
                loss = -1 * log_likelihood + kld + kld_additional
                return loss, log_likelihood, kld, kld_additional

class iwae_loss(nn.Module):
    def __init__(self):
        super(iwae_loss, self).__init__()

    def forward(self, p_y_pred, target_y, posterior_z_dist, prior_z_dist):
        """
            logexpsum with respect to num_samples axis

            Args:
                p_y_pred : [num_samples, task, num_points, y_dim]
                target_y : [num_samples, task, num_points, y_dim]
                target_y : [num_samples, task, y_dim]
                prior_z_dist : [num_samples, task, y_dim]
            
            Returns:
                iwae_loss : scalar
        """

        # reconstruction error
        log_prob = p_y_pred.log_prob(target_y)
        sum_log_prob = -1 * log_prob.sum(dim=-1).sum(dim=-1).mean(dim=-1)
        
        # reguarlization error
        kld = torch.distributions.kl.kl_divergence(posterior_z_dist, prior_z_dist).sum(dim=1).mean(dim=-1) \
            if posterior_z_dist is not None and prior_z_dist is not None else 0
        
        # logexpsum
        loss = torch.logsumexp(sum_log_prob + kld, 0)

        # additional information
        mean_log_prob = sum_log_prob.mean()
        mean_kld = kld.mean() if posterior_z_dist is not None else None

        return loss, mean_log_prob, mean_kld

class loglikelihood(nn.Module):
    def __init__(self):
        super(loglikelihood, self).__init__()

    def forward(self, p_y_pred, target_y):
        log_prob = p_y_pred.log_prob(target_y)
        sum_log_prob = log_prob.sum(dim=-1).sum(dim=-1).mean()

        return sum_log_prob

class kld(nn.Module):
    def __init__(self):
        super(kld, self).__init__()

    def forward(self, posterior, prior):
        # kld
        kld = torch.distributions.kl.kl_divergence(posterior, prior).sum(dim=1).mean()

        return kld


class elbo_loss_mse(nn.Module):
    def __init__(self, model_name=None):
        super(elbo_loss_mse, self).__init__()
        
        # model name
        self.model_name = None

        # elbo_loss
        self.elbo = elbo_loss()

        
    def forward(self, p_y_dist, target_y, posterior_z_dist=None, prior_z_dist=None, **kargs):
        # elbo loss
        loss, loglikelihood, kld, _ =\
            self.elbo(p_y_dist, target_y, posterior_z_dist=None, prior_z_dist=None)

        # mse loss
        mse = F.mse_loss(p_y_dist.loc, target_y, reduction='mean')

        return loss, loglikelihood, kld, mse
        


    

        



    
