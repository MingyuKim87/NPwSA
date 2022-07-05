import os
import time
import numpy as np
import torch

# Model
from models.cnp import CNP
from models.anp import ANP, variational_ANP
from models.np import NP
from models.parts.attention import Attention

def context_target_split_trainer(x, y, num_context, num_total_point, is_test=False, **kargs):
    """
        Args:
            x : batch_size, total_data_points, x_dim
            y : batch_size, total_data_points, y_dim
            num_context : scalar (int)
            num_extra_target : scalar (int)
            is_test : use all or sample

        Returns:
            x_context, y_context, x_target, y_target
    """

    num_points = x.shape[1]
    
    # Sample locations of context and target points (for meta-train)
    locations = np.random.choice(num_points,
                                 size=num_total_point,
                                 replace=False)

    x_context = x[:, locations[:num_context], :]
    y_context = y[:, locations[:num_context], :]
    x_target = x[:, locations, :] if not is_test else x
    y_target = y[:, locations, :] if not is_test else y

    return x_context, y_context, x_target, y_target, locations[:num_context]

def set_model_parameters(model_name, config, base_config):
    # output dim
    config['OUTPUT_DIM'] = base_config["OUTPUT_DIM"]
    config['INPUT_DIM'] = base_config["INPUT_DIM"]
    
    # Hyper Parameters
    config['ENCODER_INPUT_DIM'] = base_config['INPUT_DIM'] + base_config['OUTPUT_DIM']
    config['ENCODER_LAYER_SIZES'] = [config['HIDDEN_SIZE']] * 3 
    config['DECODER_LAYER_SIZE'] = [config['HIDDEN_SIZE']] * 3

    if model_name == "NP" or model_name == "CNP":
        config['DECODER_INPUT_DIM'] = base_config['INPUT_DIM'] + config['NUM_LATENT']
    elif model_name == "ANP" or model_name.find("ANP") >= 0:
        config['DECODER_INPUT_DIM'] = base_config['INPUT_DIM'] + config['NUM_LATENT'] + config['NUM_LATENT']

    return config

def set_model(model_name, config, device):
    if model_name == "CNP":
        model = CNP(config['ENCODER_INPUT_DIM'], config['ENCODER_LAYER_SIZES'], config['NUM_LATENT'],\
            config['DECODER_INPUT_DIM'], config['DECODER_LAYER_SIZE'], config['OUTPUT_DIM'], device)
    elif model_name == "NP":
        model = NP(config['ENCODER_INPUT_DIM'], config['ENCODER_LAYER_SIZES'], config['NUM_LATENT'],\
            config['DECODER_INPUT_DIM'], config['DECODER_LAYER_SIZE'], config['OUTPUT_DIM'], device)

    elif config['ATTENTION_TYPE'] is not None:
        DIM = {'q_last_dim' : config['INPUT_DIM'], 'k_last_dim' : config['INPUT_DIM'], \
                'v_last_dim' : config['NUM_LATENT']}

        # Attention Modules
        if model_name == "ANP" or model_name == "ANP_variational":
            # Normal multihead attention
            attention = \
                Attention(device, embedding_type='mlp', layer_sizes=[config['HIDDEN_SIZE']]*2, \
                dim=DIM, att_type=config['ATTENTION_TYPE'])

        elif model_name == "ANP_log_normal" or model_name.find("ANP_log_normal") >= 0:
            attention = \
                Attention(device, embedding_type='mlp', layer_sizes=[config['HIDDEN_SIZE']]*2, \
                dim=DIM, att_type=config['ATTENTION_TYPE'], prior_type=config["PRIOR_TYPE"],
                eps=config['EPS'], training=config['RSAMPLE_TRAINING'],\
                sigma_normal_posterior=config['SIGMA_NORMAL_POSTERIOR'],\
                sigma_normal_prior=config['SIGMA_NORMAL_PRIOR'])

        elif model_name == "ANP_weibull" or model_name.find("ANP_weibull") >= 0:
            # Bayesian attention : weibull
            attention = \
                Attention(device, embedding_type='mlp', layer_sizes=[config['HIDDEN_SIZE']]*2, \
                dim=DIM, att_type=config['ATTENTION_TYPE'], prior_type=config["PRIOR_TYPE"],\
                eps=config['EPS'], training=config['RSAMPLE_TRAINING'],\
                k_weibull=config['K_WEIBULL'])
        elif model_name == "ANP_dirichlet" or model_name.find("ANP_dirichlet") >= 0:
            # Dirichlet VAE
            attention = \
                Attention(device, embedding_type='mlp', layer_sizes=[config['HIDDEN_SIZE']]*2, \
                dim=DIM, att_type=config['ATTENTION_TYPE'], prior_type=config["PRIOR_TYPE"],\
                eps=config['EPS'], beta=config['BETA'])
        else:
            attention = None

        # Models
        if model_name.find("variational") >= 0:
            # variational ANP (after attention value)
            model = variational_ANP(config['ENCODER_INPUT_DIM'], config['ENCODER_LAYER_SIZES'], config['NUM_LATENT'],\
                config['DECODER_INPUT_DIM'], config['DECODER_LAYER_SIZE'], config['OUTPUT_DIM'], attention, device)            
        else:
            # ANP
            model = ANP(config['ENCODER_INPUT_DIM'], config['ENCODER_LAYER_SIZES'], config['NUM_LATENT'],\
                config['DECODER_INPUT_DIM'], config['DECODER_LAYER_SIZE'], config['OUTPUT_DIM'], attention, device)
    else:
        assert NotImplementedError

    return model

def get_model_dir_path_config():
        # Spliting directory and file path and return directory path + "Result"
            # Usually, on test mode
        model_dir_path = os.getcwd()
        model_root_dir_path = os.path.join(model_dir_path, 'Result')
        os.makedirs(model_root_dir_path) if not os.path.isdir(model_root_dir_path) else None

        return model_root_dir_path