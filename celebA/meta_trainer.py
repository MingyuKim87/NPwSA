import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter

import shutil
from datetime import datetime
from utils import context_target_split_trainer
from helper.plot import *

class meta_celebA_trainer(object):
    def __init__(self, models, criterion, device, data_loader,\
        optimizer=None, num_epochs=None, savedir=None, \
        val_loader=None, is_tensorboard=False, \
        num_context = 100, num_extra_target=100, is_average_by_points=False, is_iwae=False, iw_samples=5, n_batches=None):

        # img size
        self.img_size = data_loader.dataset.img_size
        self.channel = data_loader.dataset.channel
        self.height = data_loader.dataset.height
        self.width = data_loader.dataset.width
        
        # Training Setting
        self.optimizer = optimizer
        self.device = device

        # Loss function
        self.criterion = criterion

        # data_loader
        self.data_loader = data_loader
        self.val_loader = val_loader

        # data point setting
        self.num_context = num_context
        self.num_extra_target = num_extra_target
        
        # Model
        self.model, self.models = self._get_models(models)
        
        # Training paraemters
        self.num_epochs = num_epochs
        self.step = 0
        self.print_freq = 200
        self.save_freq = 10000
        self.figure_freq = 3000
        self.max_step = 200000
        self.change_params_freq = 10000
        self.is_iwae = is_iwae
        self.iw_samples = iw_samples

        # Model and result save (I/O)
        self.savedir = savedir
        self.step = 0

        # result path
        self.train_result_path, self.val_result_path = \
            self._set_result_path("result_during_training.txt", "val_result_during_training.txt")
        
        # save criteria
        self.min_loss = 1e+7

        # how to display results
        self._is_average_by_points = is_average_by_points

        # n_batches
        self.n_batches = n_batches

    def _get_models(self, models):
        model_list = None
        model = None

        if isinstance(models, list):
            model_list = []
            for model in models:
                model.to(self.device)
                model_list.append(model)

        else:
            models.to(self.device)
            model = models

        return model, model_list

    def train(self, is_both_train_val=False):
        model = self.model

        try :
            # Result file path    
            for epoch in range(self.num_epochs):
                # Termination condition
                if self.step >= self.max_step:
                    break

                # re-generate mask matrices
                self.data_loader.dataset.re_generate_mask_and_data()

                if self.val_loader is not None:
                    if is_both_train_val:
                        epoch_loss, epoch_likelihood, epoch_context_likelihood, \
                            epoch_kld, epoch_kld_additional \
                            = self._epochs_with_val_loader(self.data_loader, self.val_loader)

                    else:
                        epoch_loss, epoch_likelihood, epoch_context_likelihood, \
                            epoch_kld, epoch_kld_additional \
                            = self._epochs(self.data_loader, is_train=True)

                        epoch_loss_val, epoch_likelihood_val, epoch_context_likelihood_val, _, _ \
                            = self._epochs(self.val_loader, is_train=False)

                        # save results (for train)
                        self._write_result_args(self.train_result_path, epoch, \
                            epoch_loss, epoch_likelihood, epoch_context_likelihood, epoch_kld, epoch_kld_additional)

                        # save results (for val)
                        self._write_result_args(self.val_result_path, epoch, \
                            epoch_loss_val, epoch_context_likelihood_val)
                        
                        
                else:
                    epoch_loss, epoch_likelihood, epoch_context_likelihood, \
                        epoch_kld, epoch_kld_additional \
                        = self._epochs(self.data_loader, is_train=True)

                    # save results
                    self._write_result_args(self.val_result_path, epoch, \
                        epoch_loss_val, epoch_context_likelihood_val)

                # model save (val_loss(x), train_loss(o))
                if self.min_loss >= epoch_loss or \
                    (self.step % self.save_freq==0):
                    self._model_save(epoch_loss)
                    self.min_loss = epoch_loss

            # model save at the last step
            self._model_save(loss=None)
            
        except KeyboardInterrupt:
            shutil.rmtree(self.savedir)

            
    def test(self, is_plot=True, is_plot_uncertainty=False):
        # Result file path
        temp_dir = self._make_dir(os.path.join(self.savedir, "Result"))
        test_result_path = os.path.join(temp_dir, "result_test.txt")    
        test_component_result_path = \
            os.path.join(temp_dir, "result_component_test.txt")
        
        '''
        # Loss
        loss, likelihood, likelihood_context, kld, mse \
                    = self._epochs_test(self.data_loader)
        '''

        # Plot
            # Test data set
        context_x, context_y, target_x, target_y, mu, sigma,\
        loss, likelihood, likelihood_context, kld, mse, attention_weights_dict, reps_dict \
            = self._epochs_test(self.data_loader) 

        if self.val_loader is not None:
            context_x_val, context_y_val, target_x_val, target_y_val, mu_val, sigma_val,\
            loss_val, likelihood_val, likelihood_context_val, kld_val, mse_val, \
            attention_weights_dict_val, reps_dict_val \
                = self._epochs_test(self.data_loader)

        # Plot and save results (void function)
        self._plot_and_write_result(test_result_path, test_component_result_path, context_x, context_y, target_x, target_y, mu, sigma,\
            loss, likelihood, likelihood_context, kld, mse, attention_weights_dict, reps_dict, is_plot, is_plot_uncertainty)

        # Get weight histogram
        summary_dict = self.get_weight_histogram()

        # Add embedding 
        if self.val_loader is not None:
            self.add_embedding_summary(summary_dict, reps_dict, reps_dict_val)

    
    def test_several_context(self, context_list, attr=None):
        '''
            Args:
                context_list : a sequence of context points
                attr : celebA att
        '''
        # attr
        if attr is None:
            attr = 'All'

        # Now
        now = datetime.now()
        currentdate = now.strftime("%Y%m%d%H%M%S")
        
        # Plot
            # Test data set
        context_x_set, context_y_set, target_x_set, target_y_set, mu_set, sigma_set,\
        loss_set, likelihood_set, likelihood_context_set, kld_set, mse_set, attention_weights_dict_set, reps_dict_set \
            = self._epochs_test_by_several_contexts(self.data_loader, context_list) 

        for context_count in context_list:
            # Result file path
            temp_dir = self._make_dir(os.path.join(self.savedir, attr, "Result", currentdate))
            test_result_path = os.path.join(temp_dir, "result_test_{}.txt".format(context_count))
            test_component_result_path = os.path.join(temp_dir, "result_component_test.txt".format(context_count))

            # Filtering
            context_x = context_x_set[context_count]
            context_y = context_y_set[context_count]
            target_x = target_x_set[context_count]
            target_y = target_y_set[context_count]
            mu = mu_set[context_count]
            sigma = sigma_set[context_count]
            
            loss = loss_set[context_count]
            likelihood = likelihood_set[context_count]
            likelihood_context = likelihood_context_set[context_count]
            kld = kld_set[context_count]
            mse = mse_set[context_count]
            attention_weights_dict = attention_weights_dict_set[context_count]
            reps_dict = reps_dict_set[context_count]

            # plot and save results
            self._plot_and_write_result(temp_dir, context_x, context_y, target_x, target_y, mu, sigma,\
                loss, likelihood, likelihood_context, kld, mse, attention_weights_dict, reps_dict, \
                context_count=context_count, is_plot_uncertainty=True)


    def _epochs(self, data_loader, is_train=True):
        # total_num_points
        num_total_datapoints = data_loader.dataset.num_point
        
        # epoch 
        epoch_loss = 0.
        epoch_ll = 0.
        epoch_context_ll = 0.
        epoch_kld = 0.
        epoch_kld_additional = 0.

        for i, data in enumerate(data_loader):
            if self.n_batches is not None:
                if i == self.n_batches:
                    break
            
            # data
            mask, x, y = data
            
            # num_context, num_target
            num_context = np.random.randint(low=self.num_context, high=(num_total_datapoints - (self.num_context + self.num_extra_target)))
            num_extra_target = np.random.randint(low=self.num_extra_target, high=(num_total_datapoints - num_context))
            
            num_total_point = num_context + num_extra_target if is_train \
                else num_total_datapoints

            # Split context and target
            context_x, context_y, target_x, target_y, _ = \
                context_target_split_trainer(
                    x = x, 
                    y = y,
                    num_context = num_context,
                    num_total_point = num_total_point,
                )

            # allocate the device
            context_x, context_y, target_x, target_y = \
                context_x.to(self.device), context_y.to(self.device), \
                target_x.to(self.device), target_y.to(self.device)

            if self.is_iwae:
                # 뻥튀기
                    # x
                context_x = torch.unsqueeze(context_x, dim=0).repeat(self.iw_samples, 1, 1, 1)
                target_x = torch.unsqueeze(target_x, dim=0).repeat(self.iw_samples, 1, 1, 1)
                    # y
                context_y = torch.unsqueeze(context_y, dim=0).repeat(self.iw_samples, 1, 1, 1)
                target_y = torch.unsqueeze(target_y, dim=0).repeat(self.iw_samples, 1, 1, 1)

            # Feed forward
            p_y_pred, posterior, prior, attention_weights, reps = \
                self.model(context_x, context_y, target_x, num_total_point, target_y)

            # kld additional (bayesian attention)
            kld_additional = self.model.kld_additional \
                if hasattr(self.model, "kld_additional") else None
            
            # Evaluate loss function
            if self.is_iwae:
                loss, log_p, kld, kld_additional = self.criterion(p_y_pred, \
                    target_y, \
                    posterior,\
                    prior,\
                    kld_additional=kld_additional,\
                    is_iwae=True)
            else:
                loss, log_p, kld, kld_additional = self.criterion(p_y_pred, \
                    target_y, \
                    posterior,\
                    prior,\
                    kld_additional=kld_additional)

            # Define normal dist for context_dataset
            p_y_pred_context = torch.distributions.Normal(\
                p_y_pred.loc[:, :num_context, :], 
                p_y_pred.scale[:, :num_context, :]
            )

            # context likelihood
            log_p_context = self.criterion.log_likelihood(\
                p_y_pred_context, context_y)

            # loss
            epoch_loss += loss.item()
            epoch_ll += log_p.item()
            epoch_context_ll += log_p_context.item()
            epoch_kld += kld.item() if kld is not None else 0
            epoch_kld_additional += kld_additional.item() \
                if kld_additional is not None else 0

            # training
            if is_train:
                # Initialize optimizer
                self.optimizer.zero_grad()

                # calculate backward()
                loss.backward()
                
                # update parameters
                self.optimizer.step()

                # count steps
                self.step += 1

                # print (not divide into epoch size)
                if self.step % self.print_freq == 0:    
                    result_dict = self._organize_result(\
                        iteration=self.step, loss=loss, \
                        NLL=log_p, KLD=kld, \
                        KLD_attention=kld_additional
                    )   

                    self._print_result(**result_dict)

        if self.n_batches is not None:
            return epoch_loss / self.n_batches, \
                epoch_ll / self.n_batches, \
                epoch_context_ll / self.n_batches, \
                epoch_kld / self.n_batches, \
                epoch_kld_additional / self.n_batches
        else:
            return epoch_loss / len(data_loader), \
                    epoch_ll / len(data_loader), \
                    epoch_context_ll / len(data_loader), \
                    epoch_kld / len(data_loader), \
                    epoch_kld_additional / len(data_loader)

    def _epochs_test(self, data_loader):
        # total_num_points
        num_total_points = data_loader.dataset.num_point
        
        # container : loss
        epoch_loss = {}
        epoch_ll = {}
        epoch_ll_context = {}
        epoch_kld = {}
        epoch_mse = {}

        # container : plot
        pred_mu = {}
        pred_sigma = {}
        attention_weights_dict = {}
        reps_dict = {}

        for i, data in enumerate(data_loader):
            # data
            mask, x, y = data
            
            # num_context, num_target (num_total_point - num_context)
            num_context = np.random.randint(low=self.num_context, high=(num_total_points - (self.num_context + self.num_extra_target)))
            #num_extra_target = np.random.randint(low=self.num_extra_target, high=(num_total_datapoints - num_context))
            #num_total_point = num_total_datapoints

            # Split context and target (locations : np.array)
            context_x, context_y, target_x, target_y, _ = \
                context_target_split_trainer(
                    x = x, 
                    y = y,
                    num_context = num_context,
                    num_total_point = num_total_points,
                    is_test=True
                )
                
            # allocate the device
            context_x, context_y, target_x, target_y = \
                context_x.to(self.device), context_y.to(self.device), \
                target_x.to(self.device), target_y.to(self.device)

            for model in self.models:      
                # Feed forward
                p_y_pred, posterior, prior, attention_weights, reps = \
                    model(context_x, context_y, target_x, num_total_points, target_y)

                # Evaluate loss function (targets)
                loss, log_p, kld, mse = self.criterion(p_y_pred, target_y, posterior, prior)

                # Define normal dist for context_dataset
                p_y_pred_context = torch.distributions.Normal(\
                    p_y_pred.loc[:,:num_context,:], 
                    p_y_pred.scale[:, :num_context, :]
                )
                
                # Evaluate loss function (contexts)
                log_p_context = self.criterion.elbo.log_likelihood(\
                    p_y_pred_context, context_y)

                # loss
                if not self._is_average_by_points:
                    if i == 0:
                        epoch_loss[model._name] = loss.item()
                        epoch_ll[model._name] = log_p.item()
                        epoch_ll_context[model._name] = log_p_context.item()
                        epoch_kld[model._name] = kld.item() if kld is not None else 0
                        epoch_mse[model._name] = mse.item() \
                            if mse is not None else 0
                    else:
                        epoch_loss[model._name] += loss.item()
                        epoch_ll[model._name] += log_p.item()
                        epoch_ll_context[model._name] += log_p_context.item()
                        epoch_kld[model._name] += kld.item() if kld is not None else 0
                        epoch_mse[model._name] += mse.item() \
                            if mse is not None else 0

                else:
                    if i == 0:
                        epoch_loss[model._name] = loss.item() / num_total_points
                        epoch_ll[model._name] = log_p.item() / num_total_points
                        epoch_ll_context[model._name] = log_p_context.item() / num_total_points
                        epoch_kld[model._name] = kld.item() / num_total_points \
                            if kld is not None else 0
                        epoch_mse[model._name] = mse.item() / num_total_points \
                            if mse is not None else 0
                    
                    
                    epoch_loss[model._name] += loss.item() / num_total_points
                    epoch_ll[model._name] += log_p.item() / num_total_points
                    epoch_ll_context[model._name] += log_p_context.item() / num_context
                    epoch_kld[model._name] += kld.item() if kld is not None else 0
                    epoch_mse[model._name] += mse.item() \
                        if mse is not None else 0

                
                # Pred
                pred_mu[model._name] = p_y_pred.loc.detach().cpu().numpy()
                pred_sigma[model._name] = p_y_pred.scale.detach().cpu().numpy()
                attention_weights_dict[model._name] = attention_weights.detach().cpu().numpy() \
                    if attention_weights is not None else None
                reps_dict[model._name] = [rep.detach().cpu().numpy() for rep in reps] \
                    if type(reps) is tuple else reps.detach().cpu().numpy()

        # loss aggregated by epochs
        for (k1,v1), (k2, v2), (k3, v3), (k4, v4), (k5, v5) in \
            zip(epoch_loss.items(), epoch_ll.items(), epoch_ll_context.items(), \
                epoch_kld.items(), epoch_mse.items()):

                # Aggregation
                epoch_loss[k1] = v1 / len(data_loader)
                epoch_ll[k2] = v2 / len(data_loader)
                epoch_ll_context[k3] = v3 / len(data_loader)
                epoch_kld[k4] = v4 / len(data_loader)
                epoch_mse[k5] = v5 / len(data_loader)

        # Returns
        #if is_plot:
        context_x, context_y, target_x, target_y = \
            self._detach_gpus(
                context_x,\
                context_y,\
                target_x,\
                target_y)
        return context_x, context_y, target_x, target_y, pred_mu, pred_sigma, epoch_loss, \
            epoch_ll, epoch_ll_context, epoch_kld, epoch_mse, attention_weights_dict, reps_dict
        
        # else:
            # return epoch_loss, epoch_ll, epoch_ll_context, epoch_kld, epoch_mse

    def _epochs_with_val_loader(self, data_loader, val_dataloader):
        """
            epoch functions have both train and val loader
                have write result functions
        """

        # total_num_points
        num_total_datapoints = data_loader.dataset.num_point
        
        # epoch 
        epoch_loss = 0.
        epoch_ll = 0.
        epoch_context_ll = 0.
        epoch_kld = 0.
        epoch_kld_additional = 0.

        for i, (data, val_data) in enumerate(zip(data_loader, val_dataloader)):
            # data
            mask, x, y = data
            mask, val_x, val_y = val_data
            
            # num_context, num_target
            num_context = np.random.randint(low=self.num_context, high=(num_total_datapoints - (self.num_context + self.num_extra_target)))
            num_extra_target = np.random.randint(low=self.num_extra_target, high=(num_total_datapoints - num_context))
            num_total_point = num_context + num_extra_target

            # Split context and target
            context_x, context_y, target_x, target_y, _ = \
                context_target_split_trainer(
                    x = x, 
                    y = y,
                    num_context = num_context,
                    num_total_point = num_total_point,
                )

            # val dataset split
            val_context_x, val_context_y, val_target_x, val_target_y, _ = \
                context_target_split_trainer(
                    x = val_x, 
                    y = val_y,
                    num_context = num_context,
                    num_total_point = num_total_point,
                )


            # allocate the device
            context_x, context_y, target_x, target_y = \
                context_x.to(self.device), context_y.to(self.device), \
                target_x.to(self.device), target_y.to(self.device)

            val_context_x, val_context_y, val_target_x, val_target_y = \
                val_context_x.to(self.device), val_context_y.to(self.device), \
                val_target_x.to(self.device), val_target_y.to(self.device)

            if self.is_iwae:
                # 뻥튀기
                    # x
                context_x = torch.unsqueeze(context_x, dim=0).repeat(self.iw_samples, 1, 1, 1)
                target_x = torch.unsqueeze(target_x, dim=0).repeat(self.iw_samples, 1, 1, 1)
                    # y
                context_y = torch.unsqueeze(context_y, dim=0).repeat(self.iw_samples, 1, 1, 1)
                target_y = torch.unsqueeze(target_y, dim=0).repeat(self.iw_samples, 1, 1, 1)

                # 뻥튀기
                    # x
                val_context_x = torch.unsqueeze(val_context_x, dim=0).repeat(self.iw_samples, 1, 1, 1)
                val_target_x = torch.unsqueeze(val_target_x, dim=0).repeat(self.iw_samples, 1, 1, 1)
                    # y
                val_context_y = torch.unsqueeze(val_context_y, dim=0).repeat(self.iw_samples, 1, 1, 1)
                val_target_y = torch.unsqueeze(val_target_y, dim=0).repeat(self.iw_samples, 1, 1, 1)

            # Feed forward
            p_y_pred, posterior, prior, attention_weights, reps = \
                self.model(context_x, context_y, target_x, num_total_point, target_y)

            # Feed forward (for val)
            val_p_y_pred, val_posterior, val_prior, val_attention_weights, val_reps = \
                self.model(val_context_x, val_context_y, val_target_x, num_total_point, val_target_y)

            # kld additional (bayesian attention)
            kld_additional = self.model.kld_additional \
                if hasattr(self.model, "kld_additional") else None
            
            # Evaluate loss function
            if self.is_iwae:
                loss, log_p, kld, kld_additional = self.criterion(p_y_pred, \
                    target_y, \
                    posterior,\
                    prior,\
                    kld_additional=kld_additional,\
                    is_iwae=True)

                val_loss, val_log_p, val_kld, val_kld_additional = self.criterion(val_p_y_pred, \
                    val_target_y, \
                    val_posterior, \
                    val_prior, \
                    kld_additional=kld_additional, \
                    is_iwae=True)
            else:
                loss, log_p, kld, kld_additional = self.criterion(p_y_pred, \
                    target_y, \
                    posterior,\
                    prior,\
                    kld_additional=kld_additional)

                val_loss, val_log_p, val_kld, val_kld_additional = self.criterion(val_p_y_pred, \
                    val_target_y, \
                    val_posterior, \
                    val_prior, \
                    kld_additional=kld_additional)



            # Evaluate loss function
            if self.is_iwae:
                # Define normal dist for context_dataset
                p_y_pred_context = torch.distributions.Normal(\
                    p_y_pred.loc[:, :, :num_context, :], 
                    p_y_pred.scale[:, :, :num_context, :]
                )

                # Define normal dist for context_dataset (val)
                val_p_y_pred_context = torch.distributions.Normal(\
                    val_p_y_pred.loc[:, :, :num_context, :], 
                    val_p_y_pred.scale[:, :, :num_context, :]
                )
                
                
                
                
                # context likelihood
                log_p_context, _, _ = self.criterion.iwae_loss(\
                    p_y_pred_context, context_y, posterior, prior)

                val_log_p_context, _, _ = self.criterion.iwae_loss(\
                    val_p_y_pred_context, val_context_y, val_posterior, val_prior)

            else:
                # Define normal dist for context_dataset
                p_y_pred_context = torch.distributions.Normal(\
                    p_y_pred.loc[:, :num_context, :], 
                    p_y_pred.scale[:, :num_context, :]
                )

                # Define normal dist for context_dataset (val)
                val_p_y_pred_context = torch.distributions.Normal(\
                    val_p_y_pred.loc[:, :num_context, :], 
                    val_p_y_pred.scale[:, :num_context, :]
                )
                
                
                
                # context likelihood
                log_p_context = self.criterion.log_likelihood(\
                    p_y_pred_context, context_y)

                val_log_p_context = self.criterion.log_likelihood(\
                    val_p_y_pred_context, val_context_y)
            

            # loss
            epoch_loss += loss.item()
            epoch_ll += log_p.item()
            epoch_context_ll += log_p_context.item()
            epoch_kld += kld.item() if kld is not None else 0
            epoch_kld_additional += kld_additional.item() \
                if kld_additional is not None else 0

            # training
            
            # Initialize optimizer
            self.optimizer.zero_grad()

            # calculate backward()
            loss.backward()
                
            # update parameters
            self.optimizer.step()

            # count steps
            self.step += 1

            # print (not divide into epoch size)
            if self.step % self.print_freq == 0:    
                result_dict = self._organize_result(\
                    iteration=self.step, loss=loss, \
                    NLL=log_p, KLD=kld, \
                    KLD_attention=kld_additional
                )   

                self._print_result(**result_dict)

            if self.step % self.print_freq == 0:
                # save result (val)
                self._write_result_args(self.train_result_path, self.step, \
                    loss.item(), log_p.item(), log_p_context.item(),\
                    kld.item() if kld is not None else 0,\
                    kld_additional.item() if kld_additional is not None else 0)
        
                # save results (train)
                self._write_result_args(self.val_result_path, self.step, \
                    val_loss.item(), val_log_p.item(), val_log_p_context.item())
                

        if self.n_batches is not None:
            return epoch_loss / self.n_batches, \
                epoch_ll / self.n_batches, \
                epoch_context_ll / self.n_batches, \
                epoch_kld / self.n_batches, \
                epoch_kld_additional / self.n_batches
        else:
            return epoch_loss / len(data_loader), \
                    epoch_ll / len(data_loader), \
                    epoch_context_ll / len(data_loader), \
                    epoch_kld / len(data_loader), \
                    epoch_kld_additional / len(data_loader)


    def _epochs_test_by_several_contexts(self, data_loader, num_context_list):
        # total_num_points
        num_total_points = data_loader.dataset.num_point

        # Given data
        context_x_set = {}
        target_x_set = {}
        context_y_set = {}
        target_y_set = {}
        
        # containers based on context_data_set
        loss_by_context = {}
        ll_by_contexts = {}
        context_ll_by_contexts = {}
        kld_by_contexts = {}
        mse_by_contexts = {}

        # container : plot
        pred_mu_by_contexts = {}
        pred_sigma_by_contexts = {}
        attention_weights_dict_by_contexts = {}
        reps_dict_by_contexts = {}

        for context_count in num_context_list:
            # container : loss
            epoch_loss = {}
            epoch_ll = {}
            epoch_ll_context = {}
            epoch_kld = {}
            epoch_mse = {}

            # container : plot
            pred_mu = {}
            pred_sigma = {}
            attention_weights_dict = {}
            reps_dict = {}

            for i, data in enumerate(data_loader):
                # data
                mask, x, y = data
                
                # num_context = np.random.randint(low=context_count, high=(num_total_points - (self.num_context + self.num_extra_target)))
                num_context = context_count
                
                # Split context and target (locations : np.array)
                context_x, context_y, target_x, target_y, context_locations = \
                    context_target_split_trainer(
                        x = x, 
                        y = y,
                        num_context = num_context,
                        num_total_point = num_total_points,
                        is_test=True
                    )
                    
                # allocate the device
                context_x, context_y, target_x, target_y = \
                    context_x.to(self.device), context_y.to(self.device), \
                    target_x.to(self.device), target_y.to(self.device)

                for model in self.models:      
                    # Feed forward
                    p_y_pred, posterior, prior, attention_weights, reps = \
                        model(context_x, context_y, target_x, num_total_points, target_y)

                    # Evaluate loss function (targets)
                    loss, log_p, kld, mse = self.criterion(p_y_pred, target_y, posterior, prior)

                    # Define normal dist for context_dataset
                    p_y_pred_context = torch.distributions.Normal(\
                        # p_y_pred.loc[:,:num_context,:], 
                        # p_y_pred.scale[:, :num_context, :]
                        p_y_pred.loc[:,context_locations,:], 
                        p_y_pred.scale[:, context_locations, :]
                    )
                    
                    # Evaluate loss function (contexts)
                    log_p_context = self.criterion.elbo.log_likelihood(\
                        p_y_pred_context, context_y)

                    # loss
                    if not self._is_average_by_points:
                        if i == 0:
                            epoch_loss[model._name] = loss.item()
                            epoch_ll[model._name] = log_p.item()
                            epoch_ll_context[model._name] = log_p_context.item()
                            epoch_kld[model._name] = kld.item() if kld is not None else 0
                            epoch_mse[model._name] = mse.item() \
                                if mse is not None else 0
                        else:
                            epoch_loss[model._name] += loss.item()
                            epoch_ll[model._name] += log_p.item()
                            epoch_ll_context[model._name] += log_p_context.item()
                            epoch_kld[model._name] += kld.item() if kld is not None else 0
                            epoch_mse[model._name] += mse.item() \
                                if mse is not None else 0

                    else:
                        if i == 0:
                            epoch_loss[model._name] = loss.item() / num_total_points
                            epoch_ll[model._name] = log_p.item() / num_total_points
                            epoch_ll_context[model._name] = log_p_context.item() / num_total_points
                            epoch_kld[model._name] = kld.item() / num_total_points \
                                if kld is not None else 0
                            epoch_mse[model._name] = mse.item() / num_total_points \
                                if mse is not None else 0
                        
                        
                        epoch_loss[model._name] += loss.item() / num_total_points
                        epoch_ll[model._name] += log_p.item() / num_total_points
                        epoch_ll_context[model._name] += log_p_context.item() / num_context
                        epoch_kld[model._name] += kld.item() if kld is not None else 0
                        epoch_mse[model._name] += mse.item() \
                            if mse is not None else 0

                    
                    # Pred
                    pred_mu[model._name] = p_y_pred.loc.detach().cpu().numpy()
                    pred_sigma[model._name] = p_y_pred.scale.detach().cpu().numpy()
                    attention_weights_dict[model._name] = attention_weights.detach().cpu().numpy() \
                        if attention_weights is not None else None
                    reps_dict[model._name] = [rep.detach().cpu().numpy() for rep in reps] \
                        if type(reps) is tuple else reps.detach().cpu().numpy()

            # loss aggregated by epochs
            for (k1,v1), (k2, v2), (k3, v3), (k4, v4), (k5, v5) in \
                zip(epoch_loss.items(), epoch_ll.items(), epoch_ll_context.items(), \
                    epoch_kld.items(), epoch_mse.items()):

                    # Aggregation
                    epoch_loss[k1] = v1 / len(data_loader)
                    epoch_ll[k2] = v2 / len(data_loader)
                    epoch_ll_context[k3] = v3 / len(data_loader)
                    epoch_kld[k4] = v4 / len(data_loader)
                    epoch_mse[k5] = v5 / len(data_loader)

            context_x, context_y, target_x, target_y = \
                self._detach_gpus(
                    context_x,\
                    context_y,\
                    target_x,\
                    target_y)

            # Append
            context_x_set[context_count] = context_x
            target_x_set[context_count] = target_x
            context_y_set[context_count] = context_y
            target_y_set[context_count] = target_y
            
            loss_by_context[context_count] = epoch_loss
            ll_by_contexts[context_count] = epoch_ll
            context_ll_by_contexts[context_count] = epoch_ll_context
            kld_by_contexts[context_count] = epoch_kld
            mse_by_contexts[context_count] = epoch_mse

            pred_mu_by_contexts[context_count] = pred_mu
            pred_sigma_by_contexts[context_count] = pred_sigma
            attention_weights_dict_by_contexts[context_count] = attention_weights_dict
            reps_dict_by_contexts[context_count] = reps_dict
            

        return context_x_set, context_y_set, target_x_set, target_y_set, pred_mu_by_contexts, pred_sigma_by_contexts, loss_by_context, \
            ll_by_contexts, context_ll_by_contexts, kld_by_contexts, mse_by_contexts, attention_weights_dict_by_contexts, reps_dict_by_contexts


    def _plot_and_write_result(self, root_path,\
        context_x, context_y, target_x, target_y, mu, sigma,\
        loss, likelihood, likelihood_context, kld, mse, attention_weights_dict, reps_dict, \
        is_plot=True, is_plot_uncertainty=False, context_count=None):

        # plotting file root directory
        figure_root_directory = self._make_dir(os.path.join(root_path, "Figures"))

        # result file path
        test_result_path = os.path.join(root_path, "result_test_{}.txt".format(context_count))
        test_component_result_path = os.path.join(root_path, "result_component_test.txt".format(context_count))

        # random task index
        idx = np.random.choice(context_x.shape[0]) \
            if context_x.shape[0] != 1 else 0

        # Type cast : np.array to torch.Tensor
        context_x_tensor = torch.Tensor(context_x)
        context_y_tensor = torch.Tensor(context_y)
        target_x_tensor = torch.Tensor(target_x)
        target_y_tensor = torch.Tensor(target_y)

        # is plot
        if is_plot:
            # make given data image frames
            context_imgs = xy_to_img(context_x_tensor, context_y_tensor, self.img_size)
            target_imgs = xy_to_img(target_x_tensor, target_y_tensor, self.img_size)

            if is_plot_uncertainty:
                # Plot all images iteratively
                for i, (context_img, target_img) in enumerate(zip(context_imgs, target_imgs)):
                    # plot target img
                    plot_celebA_img(context_img, is_save=True, \
                            result_path=figure_root_directory, \
                            file_name='/celebA_context_img_{}_{}.png'.format(context_count, i))
                
                    # plot target img
                    plot_celebA_img(target_img, is_save=True, \
                            result_path=figure_root_directory, \
                            file_name='/celebA_target_img_{}_{}.png'.format(context_count, i))
            else:
                # Plot all images at once
                plot_celebA_imgs(context_imgs, is_save=True, \
                    result_path=figure_root_directory, file_name='/celebA_context_img_{}.png'.format(context_count))
            
                plot_celebA_imgs(target_imgs, is_save=True, \
                    result_path=figure_root_directory, file_name='/celebA_target_img_{}.png'.format(context_count))

        # save results
        for (k1, v1), (k2, v2), (k3, v3) \
            in zip(mu.items(), sigma.items(), attention_weights_dict.items()):

            # type cast : mu
            v1_tensor = torch.tensor(v1)

            # type cast : sigma
            v2_tensor = torch.tensor(v2)
        
            if is_plot:
                # get image tensor
                predicted_imgs = xy_to_img(target_x_tensor, v1_tensor, self.img_size)
                predicted_uncertainty = xy_to_img(target_x_tensor, v2_tensor, self.img_size, normalize_type='x')

                if is_plot_uncertainty:
                    for i, (img, uncertainty_img) in enumerate(zip(predicted_imgs, predicted_uncertainty)):
                        # plot mu
                        plot_celebA_img(img, is_save=True, \
                            result_path=figure_root_directory, \
                            file_name='/celebA_predicted_img_{}_{}_{}.png'.format(k1, context_count, i))
                    
                        # plot sigma
                            # sigma should be transformed to [C, H, W] -> [H, W]
                        uncertainty_img = uncertainty_img.mean(dim=0)
                        plot_celebA_img(uncertainty_img, is_save=True, \
                            result_path=figure_root_directory,\
                            file_name='/celebA_predicted_uncertainty_{}_{}_{}.png'.format(k1, context_count, i), is_scale_bar=True)

                # plot only mu
                plot_celebA_imgs(predicted_imgs, is_save=True, \
                    result_path=figure_root_directory, file_name='/celebA_predicted_img_{}_{}.png'.format(k1, context_count))
            
    
                '''
                if v3 is not None:
                    plot_attention_weights_heat_map(context_x, target_x, v3,\
                        title="Attention_weight_heat_map" + "_" + k3, 
                        result_path="./Result",
                        file_name="Attention_weight_heat_map" + "_" + k2,
                        idx=idx) 
                '''

            # Print (evaluation metrics)
            self._print_result(model_name=k1, loss=loss[k1], \
                likelihood=likelihood[k1], kld=kld[k1], \
                likelihood_context = likelihood_context[k1], \
                mse=mse[k1])

            self._write_result_args(test_result_path, \
                model_name=k2, loss=loss[k2], \
                likelihood=likelihood[k2], kld=likelihood[k2],\
                likelihood_context = likelihood_context[k1], \
                mse=mse[k1])

            self._write_components_args(test_component_result_path, \
                model_name = k1, target_x = target_x, target_y = target_y,\
                mu = mu[k1], sigma = sigma[k1]
            )
 
    def _detach_gpus(self, *args):
        temp = []
        for tensor in args:
            if tensor is not None:
                tensor = tensor.detach().cpu().numpy()
                temp.append(tensor)
        return tuple(temp)

    def _organize_result(self, **kargs):
        dicts = {}

        for key, value in kargs.items():
            if value is None:
                continue
            else:
                if torch.is_tensor(value):
                    dicts[key] = value.item()
                else:
                    dicts[key] = value
        return dicts

    def _print_result(self, **kargs):
        strings = ''

        for key, value in kargs.items():
            if isinstance(value, float):
                temp = "{} : {:.3f}".format(key, value)
            else:
                temp = "{} : {}".format(key, value)

            # Spacing
            temp = temp + " "
            strings = strings + temp

        print(strings)

    def _model_save(self, loss, file_path=None, **kargs):
        if file_path is None:
            file_path = self._file_name()
        
        if loss is not None:
            print("-"*10, "Training loss {:.3f} updated ! and save the model! (step:{})".format(\
            loss, self.step), "-"*10)
        else:
            file_path = self._file_name(is_last=True)
            print("-"*10, "Save the model! (step:{})".format(\
            loss, self.step), "-"*10)

        torch.save(self.model.state_dict(), file_path+"{}.pt".format(self.step))
            
    def _file_name(self, is_last=False):
        # Current Time / Indicate a filename
        '''
        now = datetime.now()
        currentdate = now.strftime("%Y%m%d%H%M%S")
        '''
        
        temp_dir = self._make_dir(os.path.join(self.savedir))
        currentname = "best_model"

        if not is_last:
            filename = os.path.join(temp_dir, currentname)
        else:
            filename = os.path.join(temp_dir, "last_step_model")

        return filename

    def _make_dir(self, dirpath):
        try:
            if not(os.path.isdir(dirpath)):
                os.makedirs(os.path.join(dirpath))
        except OSError as e:
            if e.errno != errno.EEXIST:
                print("Failed to create directory!!!!!")
        
        return os.path.join(dirpath)    

    def _set_result_path(self, train_result_name=None, val_result_name=None):
        temp_dir = self._make_dir(os.path.join(self.savedir))
        train_result_name = train_result_name if train_result_name is not None \
            else "result_during_training.txt"
        val_result_name = val_result_name if val_result_name is not None \
            else "val_result_during_training.txt"
        return os.path.join(temp_dir, train_result_name), os.path.join(temp_dir, val_result_name)


    def _write_result_args(self, filepath, *args, **kargs):
        with open(filepath, 'ab') as f:
            epoch_result= []
            for arg in args:
                if type(arg) == float or type(arg) == int:
                    epoch_result.append(arg)

            for key, value in kargs.items():
                if type(value) == float or type(value) == int:
                    temp = key + ":" + " {:.3f}".format(value)
                else:
                    temp = key + ":" + " {}".format(value)
                
                epoch_result.append(temp)

            epoch_result = [epoch_result]

            if isinstance(epoch_result[0][0], float):
                np.savetxt(f, epoch_result, delimiter=',', fmt='%.3f')
            else:
                np.savetxt(f, epoch_result, delimiter=',', fmt='%s')

        f.close()

    def _write_components_args(self, filepath, *args, **kargs):
        # values
        model_name = kargs.get("model_name", None)
        target_x = kargs.get("target_x", None)
        target_y = kargs.get("target_y", None)
        mu = kargs.get("mu", None)
        sigma = kargs.get("sigma", None)

        # assert
        assert model_name is not None or target_x is not None or \
            target_y is not None or mu is not None or sigma is not None, \
                "Specify values"

        # squeeze
        target_x, target_y, mu, sigma = \
            np.squeeze(target_x), np.squeeze(target_y), \
                np.squeeze(mu), np.squeeze(sigma)

        # target_x, y
        with open(filepath, 'ab') as f:
            for v1, v2, v3, v4 in zip(target_x, target_y, mu, sigma):  
                result = [[model_name, v1, v2, v3, v4]]
                np.savetxt(f, result, delimiter=',', fmt="%s")

        f.close()

    def get_weight_histogram(self):
        summary_container = {}
        
        for model in self.models:
            summary = SummaryWriter("./Summary/" + model._name)

            for name, param in model.named_parameters():
                summary.add_histogram(name, param.detach().cpu().numpy())

            summary_container[model._name] = summary

        return summary_container

    def add_embedding_summary(self, summary_dict, reps_dict, reps_dict_val):
        for (k1, v1), (k2, v2), (k3, v3) in \
            zip(summary_dict.items(), reps_dict.items(), reps_dict_val.items()):
            if type(v2) is list:
                # test dataset
                deterministic_rep = v2[0]
                stochastic_rep = v2[1]

                # val dataset
                deterministic_rep_val = v3[0]
                stochastic_rep_val = v3[1]

                # get shape
                task_size, num_points, dim_deterministic = deterministic_rep.shape
                _, _, dim_stochastic = stochastic_rep.shape

                # reshaping (deterministic)
                deterministic_rep = deterministic_rep.reshape(-1, dim_deterministic)
                deterministic_rep_val = deterministic_rep_val.reshape(-1, dim_deterministic)

                # reshaping (stochastic)
                stochastic_rep = stochastic_rep.reshape(-1, dim_stochastic)
                stochastic_rep_val = stochastic_rep_val.reshape(-1, dim_stochastic)

                # get labels
                label_train = ["train"] * (task_size * num_points)
                label_val = ["val"] * (task_size * num_points)
                label = label_train + label_val

                # Merge
                deterministic_rep = np.vstack((deterministic_rep, deterministic_rep_val))
                stochastic_rep = np.vstack((stochastic_rep, stochastic_rep_val))

                # Put to summary
                v1.add_embedding(mat=deterministic_rep, metadata=label, tag=k2 + "deterministic")
                v1.add_embedding(mat=stochastic_rep, metadata=label, tag=k2 + "stochastic")

            else:
                # test dataset
                stochastic_rep = v2
                # val dataset
                stochastic_rep_val = v3

                # get shape
                task_size, num_points, dim_stochastic = stochastic_rep.shape

                # reshaping (stochastic)
                stochastic_rep = stochastic_rep.reshape(-1, dim_stochastic)
                stochastic_rep_val = stochastic_rep_val.reshape(-1, dim_stochastic)

                # get labels
                label_train = ["train"] * (task_size * num_points)
                label_val = ["val"] * (task_size * num_points)
                label = label_train + label_val

                # Merge
                stochastic_rep = np.vstack((stochastic_rep, stochastic_rep_val))

                # Put to summary
                v1.add_embedding(mat=stochastic_rep, metadata=label, tag=k2 + "stochastic")



                
            

    