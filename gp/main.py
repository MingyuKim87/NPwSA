from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import yaml

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import numpy as np

# trainer
from meta_trainer import meta_1d_regressor_trainer

# Dataset
from data.gp import GPData
from data.sine import SineData

# Criterion
from models.parts.criterion import elbo_loss, elbo_loss_mse

# Utils
from utils import *
from helper.args_helper import *


def train():
    #  Parser
    args = parse_args()

    # Set save model path
    save_model_dir = get_model_dir_path_args(args, model_save=True)

    # device
        # if environ supports CUDA_VISIBLE_DEVICE :
    # os.environ['CUDA_VISIBLE_DEVICES'] = args.device
    
        # (else) if environ dosen't supports CUDA_VISIBLE_DEVICE :
    device_num = int(float(args.device))
    torch.cuda.set_device(device_num)

    device = torch.device('cuda')
    
    #  Load config
    config = yaml.load(open("./config/models.yml", 'r'), Loader=yaml.SafeLoader)
    base_config = config["Base"]
    config = set_model_parameters(args.model, config[args.model], base_config)
    
    # Model
    model = set_model(args.model, config, device)

    # Data_loader
    name_list = ["train", "val"]

    # Dataset (RBF kernel, variable params)
    # datasets = [GPData(num_tasks=base_config["TRAIN_TASK_SIZE"]*10, \
    #     num_points=base_config["TRAIN_NUM_POINTS"],
    #     x_dim=base_config["INPUT_DIM"],
    #     y_dim=base_config["OUTPUT_DIM"],
    #     n_freq=None,
    #     is_random = True, 
    #     ) for _ in range(len(name_list))]

    # Dataset (RBF kernel, fixed param)
    # datasets = [GPData(num_tasks=base_config["TRAIN_TASK_SIZE"]*10, \
    #     num_points=base_config["TRAIN_NUM_POINTS"],
    #     x_dim=base_config["INPUT_DIM"],
    #     y_dim=base_config["OUTPUT_DIM"],
    #     n_freq=None,
    #     is_random = False, 
    #     ) for _ in range(len(name_list))]
    
    #Dataset (Two kernel RBF kernel, static params)
    datasets = [GPData(num_tasks=base_config["TRAIN_TASK_SIZE"]*10, \
        num_points=base_config["TRAIN_NUM_POINTS"],
        x_dim=base_config["INPUT_DIM"],
        y_dim=base_config["OUTPUT_DIM"],
        is_random = False,
        is_t_noise = False 
        ) for _ in range(len(name_list))]

    dataloaders = dict([\
        (name, DataLoader(dataset, batch_size=base_config['TRAIN_TASK_SIZE'], shuffle=True)) \
        for name, dataset in zip(name_list, datasets)]
    )

    # Criterion
    criterion = elbo_loss(args.model)

    # Set the optimizer
    if 'WEIGHT_DECAY' in config:
        optimizer = optim.Adam(model.parameters(), lr=config['LEARNING_RATE'], weight_decay=config['WEIGHT_DECAY'])
    else:
        optimizer = optim.Adam(model.parameters(), lr=config['LEARNING_RATE'])

    # Trainer
    regressor_trainer = meta_1d_regressor_trainer(model, criterion, device, \
        dataloaders['train'] ,optimizer, args.epochs, save_model_dir,\
        dataloaders['val'], False,\
        base_config['NUM_CONTEXT_POINTS'], base_config['NUM_EXTRA_TARGET_POINTS'],\
        is_iwae=base_config['IS_IWAE'],\
        iw_samples=base_config['IW_SAMPLES'])

    # Training
    regressor_trainer.train()


def test():
    #  Parser
    args = parse_args_test()

    #device
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device
    device = torch.device('cuda')
    
    # base and model Config
    test_config = yaml.load(open("./config/test.yml", 'r'), Loader=yaml.SafeLoader)
    config = yaml.load(open("./config/models.yml", 'r'), Loader=yaml.SafeLoader)
    base_config = config['Base']

    # Set save result path
    save_result_dir = get_model_dir_path_config()

    # Data_loader
    name_list = ["test"]

    # # Dataset
    datasets = [GPData(num_tasks=base_config["TEST_TASK_SIZE"]*10, \
            num_points=base_config["TEST_NUM_POINTS"], \
            x_dim=base_config["INPUT_DIM"], \
            y_dim=base_config["OUTPUT_DIM"],\
            n_freq=None
            )]

    dataloaders = dict([\
        (name, DataLoader(dataset, batch_size=1, shuffle=False)) \
        for name, dataset in zip(name_list, datasets)]
    )

    # Model container
    algos = []

    # Create the models
    for model_name, model_path in \
        zip(test_config['model_names'], test_config['model_path']):

        #  model config
        model_config = set_model_parameters(model_name, config[model_name], base_config)
        model = set_model(model_name, model_config, device)

        # model setting (num_actions and initial_pulls (warm-up))
        model.set_name(model_name)
        model.set_device(device)
        model.to(device)

        #  check point
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint)
        algos.append(model)

    
    # Criterion
    criterion = elbo_loss_mse()
    
    # Trainer
    regressor_tester = meta_1d_regressor_trainer(algos, criterion, device, \
        dataloaders['test'],\
        savedir=save_result_dir,
        val_loader=dataloaders['val'],
        num_context=base_config['NUM_CONTEXT_POINTS'],\
        num_extra_target=dataloaders['test'].dataset._num_points - \
            base_config['NUM_CONTEXT_POINTS'],
        is_average_by_points=base_config['IS_AVERAGE_BY_POINTS']
    )

    # Test
    regressor_tester.test()
    
if __name__ == "__main__":
    train()

    



    




