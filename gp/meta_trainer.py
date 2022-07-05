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
from helper.plot import plot_1D_regression, plot_attention_weights_heat_map

class meta_1d_regressor_trainer(object):
    def __init__(self, models, criterion, device, data_loader,\
        optimizer=None, num_epochs=None, savedir=None, \
        val_loader=None, is_tensorboard=False, \
        num_context = 3, num_extra_target=5, is_average_by_points=False, is_iwae=False, iw_samples=5):

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
        
        # save criteria
        self.min_loss = 1e+7

        # how to display results
        self._is_average_by_points = is_average_by_points


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

    def train(self):
        model = self.model

        try :
            # Result file path
            temp_dir = self._make_dir(os.path.join(self.savedir))
            train_result_path = os.path.join(temp_dir, "result_during_training.txt")
                
            for epoch in range(self.num_epochs):
                
                # re-generate data (random params)
                if self.data_loader.dataset._is_random and \
                    self.step % self.change_params_freq == 0:
                    self.data_loader.dataset.update_data()

                # Termination condition
                if self.step >= self.max_step:
                    break
                
                epoch_loss, epoch_likelihood, epoch_kld, epoch_kld_additional \
                    = self._epochs(self.data_loader, is_train=True)

                self._write_result_args(train_result_path, epoch, \
                    epoch_loss, epoch_likelihood, epoch_kld, epoch_kld_additional)

                if self.val_loader is not None:
                    val_result_path = os.path.join(temp_dir, "val_result_during_training.txt")
                
                    epoch_loss_val, _, _, _ = self._epochs(self.val_loader, is_train=False)

                    # save results
                    self._write_result_args(val_result_path, epoch, \
                        epoch_loss_val)

                    # model save (val_loss, train_loss(x))
                    if self.min_loss >= epoch_loss or \
                        (self.step % self.save_freq==0):
                        self._model_save(epoch_loss)
                        self.min_loss = epoch_loss

            # model save at the last step
            self._model_save(loss=None)
            
        except KeyboardInterrupt:
            shutil.rmtree(self.savedir)


    def test(self):
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
            = self._epochs_test(self.data_loader, is_plot=True)

        if self.val_loader is not None:
            context_x_val, context_y_val, target_x_val, target_y_val, mu_val, sigma_val,\
            loss_val, likelihood_val, likelihood_context_val, kld_val, mse_val, \
            attention_weights_dict_val, reps_dict_val \
                = self._epochs_test(self.data_loader, is_plot=True)

        # random task index
        idx = np.random.choice(context_x.shape[0]) \
            if context_x.shape[0] != 1 else 0

        # Plot 1D regression
        for (k1, v1), (k2, v2), (k3, v3) \
            in zip(mu.items(), sigma.items(), attention_weights_dict.items()):
            
            plot_1D_regression(context_x, context_y, target_x, target_y, \
                pred_mu=v1,
                pred_sigma=v2,
                title="1D_regression" + "_" + k1,
                result_path="./Result",
                file_name="1D_regression" + "_" + k2,
                idx = idx)

            if v3 is not None:
                plot_attention_weights_heat_map(context_x, target_x, v3,\
                    title="Attention_weight_heat_map" + "_" + k3, 
                    result_path="./Result",
                    file_name="Attention_weight_heat_map" + "_" + k2,
                    idx=idx) 

            # Print
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

        # Get wegith histogram
        summary_dict = self.get_weight_histogram()

        # Add embedding 
        if self.val_loader is not None:
            self.add_embedding_summary(summary_dict, reps_dict, reps_dict_val)
        
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

            
    def _epochs(self, data_loader, is_train=True):
        epoch_loss = 0.
        epoch_ll = 0.
        epoch_kld = 0.
        epoch_kld_additional = 0.

        for _, data in enumerate(data_loader):
            # data
            x, y = data
            
            # num_context, num_target
            num_context = np.random.randint(low=self.num_context, high=(data_loader.dataset._num_points - (self.num_context + self.num_extra_target)))
            num_extra_target = np.random.randint(low=self.num_extra_target, high=(data_loader.dataset._num_points - num_context))
            
            num_total_point = num_context + num_extra_target if is_train \
                else data_loader.dataset._num_points

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

            # loss
            epoch_loss += loss.item()
            epoch_ll += log_p.item()
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

        return epoch_loss / len(data_loader), \
                epoch_ll / len(data_loader), \
                epoch_kld / len(data_loader), \
                epoch_kld_additional / len(data_loader)

    def _epochs_test(self, data_loader, is_plot=False):
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
            x, y = data
            
            # num_context, num_target (num_total_point - num_context)
            num_context = np.random.randint(low=self.num_context, high=self.num_context + 10)
            # num_context = np.random.randint(low=self.num_context, high=self.num_extra_target)
            num_total_point = data_loader.dataset._num_points

            # Split context and target (locations : np.array)
            context_x, context_y, target_x, target_y, locations = \
                context_target_split_trainer(
                    x = x, 
                    y = y,
                    num_context = num_context,
                    num_total_point = num_total_point,
                    is_test=True
                )
            # allocate the device
            context_x, context_y, target_x, target_y = \
                context_x.to(self.device), context_y.to(self.device), \
                target_x.to(self.device), target_y.to(self.device)

            for model in self.models:      
                # Feed forward
                p_y_pred, posterior, prior, attention_weights, reps = \
                    model(context_x, context_y, target_x, num_total_point, target_y)

                # Evaluate loss function (targets)
                loss, log_p, kld, mse = self.criterion(p_y_pred, target_y, posterior, prior)

                # Define normal dist for context_dataset
                p_y_pred_context = torch.distributions.Normal(\
                    p_y_pred.loc[:,locations,:], 
                    p_y_pred.scale[:, locations, :]
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
                    
                    
                    epoch_loss[model._name] += loss.item() / num_total_point
                    epoch_ll[model._name] += log_p.item() / num_total_point
                    epoch_ll_context[model._name] += log_p_context.item() / num_context
                    epoch_kld[model._name] += kld.item() if kld is not None else 0
                    epoch_mse[model._name] += mse.item() \
                        if mse is not None else 0

                if is_plot:
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
        if is_plot:
            context_x, context_y, target_x, target_y = \
                self._detach_gpus(
                    context_x,\
                    context_y,\
                    target_x,\
                    target_y)
            return context_x, context_y, target_x, target_y, pred_mu, pred_sigma, epoch_loss, \
                epoch_ll, epoch_ll_context, epoch_kld, epoch_mse, attention_weights_dict, reps_dict
        else:
            return epoch_loss, epoch_ll, epoch_ll_context, epoch_kld, epoch_mse
 
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



                
            

    