from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
from absl import app
from absl import flags
import yaml

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import numpy as np

# trainer
from meta_trainer_LVmodel import meta_1d_regressor_LV_trainer
from meta_trainer import meta_1d_regressor_trainer

# Dataset
from data.gp import GPData
from data.sine import SineData
from data.lotka_volterra import LVdata, HLdata
from data.artic import ArticData

# Criterion
from models.parts.criterion import elbo_loss, elbo_loss_mse

# Utils
from utils import *
from utils_test import *
from helper.args_helper import *


def train():
    #  Parser
    args = parse_args()

    # Set save model path
    save_model_dir = get_model_dir_path_args(args, model_save=True)

    # device
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device
    device = torch.device('cuda')
    
    #  Load config
    config = yaml.load(open("./config/models.yml", 'r'), Loader=yaml.SafeLoader)
    base_config = config["Base"]
    config = set_model_parameters(args.model, config[args.model], base_config)
    
    # Model
    model = set_model(args.model, config, device)

    # Data_loader
    dataset_root = "./data/dataset"
    # name_list = ["train", "val"]
    name_list = ["train", "val"]

    #Dataset (lotka_volterra)
    datasets = [
        LVdata(file_path=os.path.join(dataset_root, name + ".tar"),\
        is_noise=False,\
        noise_coef=100) for name in name_list
    ]

    dataloaders = dict([\
        (name, DataLoader(dataset, batch_size=1, shuffle=True)) \
        for name, dataset in zip(name_list, datasets)]
    )

    # Criterion
    criterion = elbo_loss(args.model)

    # Set the optimizer
    optimizer = optim.Adam(model.parameters(), lr=config['LEARNING_RATE'])

    # Trainer
    regressor_trainer = meta_1d_regressor_LV_trainer(model, criterion, device, \
        dataloaders['train'] ,optimizer, args.epochs, save_model_dir,\
        dataloaders['val'], \
        # None, \
        False,\
        base_config['NUM_CONTEXT_POINTS'], base_config['NUM_EXTRA_TARGET_POINTS'])

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
    dataset_root = "./data/dataset"
    
    if args.is_hare_lynx:
        name_list = ["LynxHare"]

        # Dataset
        datasets = [HLdata(file_path=os.path.join(dataset_root, name + ".txt"), \
            num_tasks=base_config["TEST_TASK_SIZE"]*10)\
             for name in name_list]

        # Dataloader
        dataloaders = dict([\
           (name, DataLoader(dataset, batch_size=base_config['TEST_TASK_SIZE'], shuffle=False)) \
            for name, dataset in zip(name_list, datasets)])
    else:
        name_list = ["train"]
        
        # Dataset
        datasets = [LVdata(file_path=os.path.join(dataset_root, name + ".tar"))\
             for name in name_list]
        

        # Dataloader
        dataloaders = dict([\
           (name, DataLoader(dataset, batch_size=1, shuffle=False)) \
            for name, dataset in zip(name_list, datasets)])


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
        dataloaders[name_list[0]],\
        savedir=save_result_dir,
        num_context=base_config['NUM_CONTEXT_POINTS'],\
        num_extra_target=base_config['NUM_EXTRA_TARGET_POINTS'],
        is_average_by_points=base_config['IS_AVERAGE_BY_POINTS']
    )

    # Test
    regressor_tester.test()

def get_attention_score():
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
    dataset_root = "./data/dataset"
    
    if args.is_hare_lynx:
        name_list = ["LynxHare"]

        # Dataset
        datasets = [HLdata(file_path=os.path.join(dataset_root, name + ".txt"), \
            num_tasks= base_config["TEST_TASK_SIZE"]*10)\
             for name in name_list]
        
        # Dataloader
        dataloaders = dict([\
           (name, DataLoader(dataset, batch_size=1, shuffle=False)) \
            for name, dataset in zip(name_list, datasets)])
    else:
        name_list = ["val"]
        
        # Dataset
        datasets = [LVdata(file_path=os.path.join(dataset_root, name + ".tar"))\
             for name in name_list]
        

        # Dataloader
        dataloaders = dict([\
           (name, DataLoader(dataset, batch_size=1, shuffle=False)) \
            for name, dataset in zip(name_list, datasets)])


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
        dataloaders[name_list[0]],\
        savedir=save_result_dir,
        num_context=base_config['NUM_CONTEXT_POINTS'],\
        num_extra_target=base_config['NUM_EXTRA_TARGET_POINTS'],
        is_average_by_points=base_config['IS_AVERAGE_BY_POINTS']
    )

    # Test
    regressor_tester.test()
    attention_score = regressor_tester.get_attention_score()

    return attention_score


def run_test(times):
    # run test
    for _ in range(times):
        test()

    # organize result statistics
    result = organize_result("./Result/result_test.txt")
    get_statistic(result, "./Result.csv")

    
if __name__ == "__main__":
    # For train
    # train()
    # For test
    run_test(5)

    
    

    



    




