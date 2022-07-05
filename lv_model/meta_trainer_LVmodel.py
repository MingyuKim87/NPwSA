import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import shutil
from datetime import datetime
from utils import context_target_split_trainer,\
     context_target_split_trainer_LVmodel
from helper.plot import plot_1D_regression

class meta_1d_regressor_LV_trainer(object):
    def __init__(self, models, criterion, device, data_loader,\
        optimizer=None, num_epochs=None, savedir=None, \
        val_loader=None, is_tensorboard=False, \
        num_context = 15, num_extra_target=15):

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
        self.figure_freq = 5000
        self.max_step = 200000
        self.is_noise = self.data_loader.dataset._is_noise

        # Model and result save (I/O)
        self.savedir = savedir
        self.step = 0
        
        # save criteria
        self.min_loss = 1e+7

    
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


    def _standardize(self, context_x, context_y, target_x, target_y):
        with torch.no_grad():
            # for X
                # Caculate parameters
            mu_x, sigma_x = context_x.mean(-2, keepdim=True), context_x.std(-2, keepdim=True)
            
                # Exceptional cases
            sigma_x[sigma_x==0] = 1.0

                # Context, target standardize
            context_x = (context_x - mu_x) / (sigma_x + 1e-5)
            target_x = (target_x - mu_x) / (sigma_x + 1e-5)

            # for Y
                # calculate parameters
            mu_y, sigma_y = context_y.mean(-2, keepdim=True), context_y.std(-2, keepdim=True)

                # Exceptional cases
            sigma_y[sigma_y==0] = 1.0
    
            context_y = (context_y - mu_y) / (sigma_y + 1e-5)
            target_y = (target_y - mu_y) / (sigma_y + 1e-5)

            return context_x, context_y, target_x, target_y, \
                (mu_x, sigma_x), (mu_y, sigma_y)

    def _reverse_standarize(self, context_x, context_y, target_x, target_y, \
        x_std_params, y_std_params):
        '''
            Get back all values based on std_params

            Args : 
                standardized context_x, context_y, target_x, target_y 
                x_std_params : (mu_x, sigma_x)
                y_std_parms : (mu_y, sigma_y)

            Returns:
                original context_x, context_y, target_x, target_y
        '''
        # params
        mu_x, sigma_x = x_std_params
        mu_y, sigma_y = y_std_params

        # get back to originals
        context_x = context_x * (sigma_x + 1e-5) + mu_x
        target_x = target_x * (sigma_x + 1e-5) + mu_x

        context_y = context_y * (sigma_y + 1e-5) + mu_y
        target_y = target_y * (sigma_y + 1e-5) + mu_y

        return context_x, context_y, target_x, target_y


    def train(self):
        model = self.model

        try :
            # Result file path
            temp_dir = self._make_dir(self.savedir)
            train_result_path = os.path.join(temp_dir, "result_during_training.txt")
                
            for epoch in range(self.num_epochs):
                
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
                if (self.min_loss >= epoch_loss) or \
                    (self.step % self.save_freq == 0):
                    self._model_save(epoch_loss)
                    self.min_loss = epoch_loss
                        
            # model save at the last step
            self._model_save(loss=None)
            
        except KeyboardInterrupt:
            shutil.rmtree(self.savedir)


    def test(self):
        # Result file path
        temp_dir = self._make_dir(self.savedir)
        test_result_path = os.path.join(temp_dir, "result_test.txt")
        test_component_result_path = \
            os.path.join(temp_dir, "result_component_test.txt")
        
        '''
        # Loss
        loss, likelihood, likelihood_context, kld, mse \
                    = self._epochs_test(self.data_loader)
        '''

        # Plot
        context_x, context_y, target_x, target_y, mu, sigma,\
        loss, likelihood, likelihood_context, kld, mse, \
        x_std_params, y_std_params, \
            = self._epochs_test(self.data_loader, is_plot=True)

        # random task index
        idx = np.random.choice(context_x.shape[0])

        # Plot 1D regression
        for (k1, v1), (k2, v2) in zip(mu.items(), sigma.items()):
            plot_1D_regression(context_x, context_y, target_x, target_y, \
                pred_mu=v1,
                pred_sigma=v2,
                title="1D_regression" + "_" + k1,
                result_path="./Result",
                file_name="1D_regression" + "_" + k2,
                idx = idx)
                

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

    def _epochs(self, data_loader, is_train=True):
        epoch_loss = 0.
        epoch_ll = 0.
        epoch_kld = 0.
        epoch_kld_additional = 0.

        for _, data in enumerate(data_loader):
            # data
            context_x, context_y, target_x, target_y = \
                context_target_split_trainer_LVmodel(data)

            # standardize
            context_x, context_y, target_x, target_y, _, _ = \
                self._standardize(context_x, context_y, target_x, target_y)

            # allocate the device
            context_x, context_y, target_x, target_y = \
                context_x.to(self.device), context_y.to(self.device), \
                target_x.to(self.device), target_y.to(self.device)
            
            # num_context, num_target
            num_context = context_x.shape[1]
            num_total_point = target_x.shape[1]
            num_extra_target = num_total_point - num_context
            
            # Feed forward
            p_y_pred, posterior, prior = \
                self.model(context_x, context_y, target_x, num_total_point, target_y)

            # kld additional (bayesian attention)
            kld_additional = self.model.kld_additional \
                if hasattr(self.model, "kld_additional") else None
            
            
            # Evaluate loss function
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
                        LL=log_p, KLD=kld, \
                        KLD_attention=kld_additional
                    )   
                    self._print_result(**result_dict)

        return epoch_loss / len(data_loader), \
                epoch_ll / len(data_loader), \
                epoch_kld / len(data_loader), \
                epoch_kld_additional / len(data_loader)

    def _epochs_test(self, data_loader, is_plot=False):
        # container : loss
        loss_models = {}
        ll_models = {}
        ll_context_models = {}
        kld_models = {}
        mse_models = {}

        # container : plot
        pred_mu = {}
        pred_sigma = {}

        for model in self.models:
            epoch_loss = 0.
            epoch_ll = 0.
            epoch_ll_context = 0.
            epoch_kld = 0.
            epoch_mse = 0.
            
            
            for _, data in enumerate(data_loader):
                # data
                context_x, context_y, target_x, target_y = \
                    context_target_split_trainer_LVmodel(data)

                # allocate the device
                context_x, context_y, target_x, target_y = \
                    context_x.to(self.device), context_y.to(self.device), \
                    target_x.to(self.device), target_y.to(self.device)

                # standardize
                context_x, context_y, target_x, target_y,\
                    x_std_params, y_std_params = \
                    self._standardize(context_x, context_y, target_x, target_y)

                # num_context, num_target
                num_context = context_x.shape[1]
                num_total_point = target_x.shape[1]
                num_extra_target = num_total_point - num_context

                
                # Feed forward
                p_y_pred, posterior, prior = \
                    model(context_x, context_y, target_x, num_total_point, target_y)

                # Feed forward for context
                p_y_pred_context, _, _ = \
                    model(context_x, context_y, context_x, num_context, context_y)

                # Evaluate loss function (targets)
                loss, log_p, kld, mse = self.criterion(p_y_pred, target_y, posterior, prior)

                # Evaluate loss function (contexts)
                log_p_context = self.criterion.elbo.log_likelihood(\
                    p_y_pred_context, context_y)

                # loss
                epoch_loss += loss.item()
                epoch_ll += log_p.item()
                epoch_ll_context += log_p_context.item()
                epoch_kld += kld.item() if kld is not None else 0
                epoch_mse += mse.item() \
                    if mse is not None else 0

            # loss
            loss_models[model._name] = epoch_loss / len(data_loader)
            ll_models[model._name] = epoch_ll / len(data_loader)
            ll_context_models[model._name] = epoch_ll_context / len(data_loader)
            kld_models[model._name] = epoch_kld / len(data_loader)
            mse_models[model._name] = epoch_mse / len(data_loader)

            if is_plot:
                # Pred
                pred_mu[model._name] = p_y_pred.loc.detach().cpu().numpy()
                pred_sigma[model._name] = p_y_pred.scale.detach().cpu().numpy()
                x_std_params = x_std_params.detach().cpu().numpy()
                y_std_params = y_std_params.detach().cpu().numpy()
            
        # Returns
        if is_plot:
            context_x, context_y, target_x, target_y = \
                self._detach_gpus(
                    context_x,\
                    context_y,\
                    target_x,\
                    target_y,)
            return context_x, context_y, target_x, target_y, pred_mu, pred_sigma, loss_models, \
                ll_models, ll_context_models, kld_models, mse_models, x_std_params, y_std_params
        else:
            return loss_models, ll_models, ll_context_models, kld_models, mse_models
 
    def _detach_gpus(self, *args):
        temp = []
        for tensor in args:
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
        
        temp_dir = self._make_dir(self.savedir)
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



                
            

    