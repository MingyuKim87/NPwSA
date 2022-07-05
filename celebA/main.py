from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import time
import yaml

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import numpy as np

# trainer
from meta_trainer import meta_celebA_trainer

# Dataset
from data.celebA_dataset import *
from data.celebA_utils import * 

# Criterion
from models.parts.criterion import elbo_loss, elbo_loss_mse

# Utils
from utils import *
from helper.args_helper import *
from helper.plot import *


def train(is_both_train_val=False):
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
    
    # file path (User defined)
    celebA_root_path = ''
    
    #  Load config
    config = yaml.load(open("./config/models.yml", 'r'), Loader=yaml.SafeLoader)
    base_config = config["Base"]
    config = set_model_parameters(args.model, config[args.model], base_config)
    
    # Model
    model = set_model(args.model, config, device)

    # Data_loader
    name_list = ["train", "val"]

    # Transform : data import
    transform = transforms.Compose([
        transforms.CenterCrop(base_config['CENTER_CROP_PIXEL']),
        transforms.Resize(base_config['IMG_SIZE'][-1]),
        transforms.ToTensor()
    ])

    # Dataset : celebA
    datasets = [CelebADataset(
        datatype = datatype,
        celebA_root_path=celebA_root_path,
        num_point=base_config["TRAIN_NUM_POINTS"],
        img_size=base_config['IMG_SIZE'],
        transform=transform,
        attr=base_config.get("ATTR_TRAIN", None),
        is_rgb=base_config.get("IS_RGB", True)
        ) for datatype in name_list]

    # Data loader
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
    image_completion_trainer = meta_celebA_trainer(model, criterion, device, \
        dataloaders['train'], optimizer, args.epochs, save_model_dir,\
        dataloaders['val'], False,\
        base_config['NUM_CONTEXT_POINTS'], base_config['NUM_EXTRA_TARGET_POINTS'],\
        is_average_by_points=base_config['IS_AVERAGE_BY_POINTS'],
        is_iwae=base_config['IS_IWAE'],\
        iw_samples=base_config['IW_SAMPLES'])

    # Training
    image_completion_trainer.train(is_both_train_val)


def test(is_several_contexts=False, num_context_list=[10, 50, 100], is_plot_training_curve=False):
    #  Parser
    args = parse_args_test()

    # device
        # if environ supports CUDA_VISIBLE_DEVICE :
    # os.environ['CUDA_VISIBLE_DEVICES'] = args.device
    
        # (else) if environ dosen't supports CUDA_VISIBLE_DEVICE :
    device_num = int(float(args.device))
    torch.cuda.set_device(device_num)
    device = torch.device('cuda')
    
    # file path (User defined)
    celebA_root_path = ''
    
    #  Load config
    test_config = yaml.load(open("./config/test.yml", 'r'), Loader=yaml.SafeLoader)
    config = yaml.load(open("./config/models.yml", 'r'), Loader=yaml.SafeLoader)
    base_config = config['Base']
    
    # Set save result path
    save_result_dir = get_model_dir_path_config()
    
    # Data_loader
    name_list = ["val"]

    # Transform : data import
    transform = transforms.Compose([
        transforms.CenterCrop(base_config['CENTER_CROP_PIXEL']),
        transforms.Resize(base_config['IMG_SIZE'][-1]),
        transforms.ToTensor()
    ])

    
    # Dataset : celebA
    datasets = [CelebADataset(
        datatype = datatype,
        celebA_root_path=celebA_root_path,
        num_point=base_config["TEST_NUM_POINTS"],
        img_size=base_config['IMG_SIZE'],
        transform=transform,
        attr=base_config.get("ATTR_TEST", None),
        is_rgb=base_config.get("IS_RGB", True)
        ) for datatype in name_list]

    
    # Data loader
    dataloaders = dict([\
        (name, DataLoader(dataset, batch_size=base_config['TEST_TASK_SIZE'], shuffle=False, drop_last=True)) \
        for name, dataset in zip(name_list, datasets)]
    )

    # Model container
    algos = []

    # Create the models
    for model_name, model_path in \
        zip(test_config['model_names'], test_config['model_path']):

        # extract model names
        model_info = extract_model_name(model_name)
        
        #  model config
        model_config = set_model_parameters(model_info, config[model_info], base_config)
        model = set_model(model_info, model_config, device)

        # model setting (num_actions and initial_pulls (warm-up))
        model.set_name(model_name)
        model.set_device(device)
        model.to(device)

        #  check point
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint)
        algos.append(model)
    
    '''
    if is_plot_training_curve:
        for model_name, train_result_path, val_result_path in \
            zip(test_config["model_names"], test_config["train_result"], test_config["val_result"]):

            plot_training_curve(train_result_path, val_result_path, \
                title= "training curve : {}".format(model_name),\
                result_path="./Result",
                file_name= "training_curve_{}.png".format(model_name)
            )
    '''
    # Criterion
    criterion = elbo_loss_mse()

    # Trainer
    image_completion_tester = meta_celebA_trainer(algos, criterion, device, \
        dataloaders['val'],\
        savedir=save_result_dir,
        val_loader = None, 
        is_tensorboard=False,\
        num_context=base_config['NUM_CONTEXT_POINTS'],\
        num_extra_target=base_config['NUM_EXTRA_TARGET_POINTS'],\
        is_average_by_points=base_config['IS_AVERAGE_BY_POINTS'],
        is_iwae=base_config['IS_IWAE'],\
        iw_samples=base_config['IW_SAMPLES'])
    
    # Test
    if is_several_contexts:
        image_completion_tester.test_several_context(context_list=num_context_list, \
            attr=base_config.get("ATTR_TEST", None))
    else:
        image_completion_tester.test(is_plot=test_config['IS_PLOT'])
    

def organize_result(file_path):

    def initialize_result(result, split_results):
        
        
        result[split_results[1]] = \
                [\
                    [split_results[3]], \
                    [split_results[5]], \
                    [split_results[7]], \
                    [split_results[9]], \
                    [split_results[11]], \
                ]

        return result

    def add_result(result, split_results):
        model_name = split_results[1]
        
        result[model_name]\
                [0].append(split_results[3])

        result[model_name]\
                [1].append(split_results[5])

        result[model_name]\
                [2].append(split_results[7])

        result[model_name]\
                [3].append(split_results[9])

        result[model_name]\
                [4].append(split_results[11])
        
        return result

    # Main algorithm
    f = open(file_path, 'r')

    # Empty set
    result = {}

    while True:
        # Read line
        line = f.readline().rstrip("\n")
        if not line: break
    
        # Split word
        split_results = re.split("[:,]", line)

        if split_results[1] in result.keys():
            result = add_result(result, split_results)
        else:
            result = initialize_result(result, split_results)
            
    f.close()
    return result

def get_statistic(result, file_path):
    '''
        Args:
            result : dict
        Returns : 
            file_statiscis
    '''

    with open(file_path, "ab") as f:

        # Head
        print_result = [["model_name", "mean_loss", "mean_ll", "std_loglikelihood", \
            "mean_context_ll", "mean_mse"]]
        np.savetxt(f, print_result, delimiter=',', fmt='%s')

        # Body
        for key, values in result.items():
            model_name = key
            mean_loss = np.asfarray(np.array(values[0]), float).mean()
            mean_ll = np.asfarray(np.array(values[1]), float).mean()
            std_loglikelihood = np.std(np.asfarray(np.array(values[1]), float))
            mean_context_ll = np.asfarray(np.array(values[3]), float).mean()
            mean_mse = np.asfarray(np.array(values[4]), float).mean()

            print_result = [[key, mean_loss, mean_ll, std_loglikelihood,\
                mean_context_ll, mean_mse]]

            np.savetxt(f, print_result, delimiter=',', fmt='%s')

        f.close()

def multiple_tests(test_time, file_path):
    
    # run test NPs
    for i in range(test_time):
        test()
    
    # organize result statistics
    result = organize_result(file_path)

    # name
    file_path_split = os.path.split(file_path)

    # rename
    new_filename = "summary_" + str(file_path_split[1])
    new_filename = new_filename.split(".")[0] + ".csv"
        
    new_file_path = os.path.join("./", new_filename)
        
    # obtain summary
    get_statistic(result, new_file_path)

    
if __name__ == "__main__":    
    # train
    # train(is_both_train_val=True)

    # test
    test(is_several_contexts=True, num_context_list=[50, 100, 150, 200, 300, 400, 500, 600])

    

    
