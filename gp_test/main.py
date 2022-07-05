from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import re
# from absl import app
# from absl import flags
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
from helper.plot import *


def train():
    #  Parser
    args = parse_args()

    # Set save model path
    save_model_dir = get_model_dir_path_args(args, model_save=True)

    #device
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device
    device = torch.device('cuda')
    
    #  Load config
    config = yaml.load(open("./config/models.yml", 'r'), Loader=yaml.SafeLoader)
    base_config = config["Base"]
    config = set_model_parameters(args.model, config[args.model], base_config)
    
    # Model
    model = set_model(args.model, config, device)

    # Data_loader
    name_list = ["train", "val"]

    # Dataset
    datasets = [GPData(num_tasks=base_config["TRAIN_TASK_SIZE"]*10, \
        num_points=base_config["TRAIN_NUM_POINTS"],
        x_dim=base_config["INPUT_DIM"],
        y_dim=base_config["OUTPUT_DIM"]) for _ in range(len(name_list))]

    dataloaders = dict([\
        (name, DataLoader(dataset, batch_size=base_config['TRAIN_TASK_SIZE'], shuffle=True)) \
        for name, dataset in zip(name_list, datasets)]
    )

    # Criterion
    criterion = elbo_loss(args.model)

    # Set the optimizer
    optimizer = optim.Adam(model.parameters(), lr=config['LEARNING_RATE'])

    # Trainer
    regressor_trainer = meta_1d_regressor_trainer(model, criterion, device, \
        dataloaders['train'] ,optimizer, args.epochs, save_model_dir,\
        dataloaders['val'], False,\
        base_config['NUM_CONTEXT_POINTS'], base_config['NUM_EXTRA_TARGET_POINTS'])

    # Training
    regressor_trainer.train()


def test(is_plot_training_curve=False):
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
    name_list = ["rbf", "rbf_random_params", "rbf_periodic", "rbf_t_noise", 'matern', 'matern_periodic',\
        'matern_t_noise', 'periodic', 'periodic_t_noise']

    
    # Dataset 
        # normal Two GP
    datasets = [
        
        # rbf
        GPData(num_tasks=base_config["TEST_TASK_SIZE"]*10, \
        num_points=base_config["TEST_NUM_POINTS"],
        x_dim=base_config["INPUT_DIM"],
        y_dim=base_config["OUTPUT_DIM"],
        n_freq=None),

        # matern
        GPData(num_tasks=base_config["TEST_TASK_SIZE"]*10, \
        num_points=base_config["TEST_NUM_POINTS"],
        x_dim=base_config["INPUT_DIM"],
        y_dim=base_config["OUTPUT_DIM"],
        kernel_type='matern',
        n_freq=None),

        # periodic
        GPData(num_tasks=base_config["TEST_TASK_SIZE"]*10, \
        num_points=base_config["TEST_NUM_POINTS"],
        x_dim=base_config["INPUT_DIM"],
        y_dim=base_config["OUTPUT_DIM"],
        kernel_type='periodic',
        n_freq=None)
    ]

        # shift and extrapolate Two GP
    # datasets = [GPData(num_tasks=base_config["TEST_TASK_SIZE"]*10, \
    #     num_points=base_config["TEST_NUM_POINTS"], \
    #     x_dim=base_config["INPUT_DIM"], \
    #     y_dim=base_config["OUTPUT_DIM"], x_minimum=-6., x_maximum=6.,\
    #     shift=np.random.randint(10) * np.random.randn()
    #     ) for _ in range(len(name_list))]

    #     # extrapolate Two GP
    # datasets = [GPData(num_tasks=base_config["TEST_TASK_SIZE"]*10, \
    #     num_points=base_config["TEST_NUM_POINTS"], \
    #     x_dim=base_config["INPUT_DIM"], \
    #     y_dim=base_config["OUTPUT_DIM"], x_minimum=-8., x_maximum=8.
    #     ) for _ in range(len(name_list))]

    #     # Sine + peridoic noise Dataset
    # datasets = [SineData(num_tasks=base_config["TRAIN_TASK_SIZE"]*10, \
    #     num_points=base_config["TRAIN_NUM_POINTS"],\
    #     x_dim =base_config["INPUT_DIM"],\
    #     y_dim=base_config["OUTPUT_DIM"],\
    #     x_range=(-4, 4),\
    #     is_noise=False) for _ in range(len(name_list))]

    # dataloaders = dict([\
    #     (name, DataLoader(dataset, batch_size=base_config['TEST_TASK_SIZE'], shuffle=False)) \
    #     for name, dataset in zip(name_list, datasets)]
    # )

    # dataloaders = dict([\
    #     (name, DataLoader(dataset, batch_size=base_config['TEST_TASK_SIZE'], shuffle=False)) \
    #     for name, dataset in zip(name_list, datasets)]
    # )

    dataloaders = dict([\
        (name, DataLoader(dataset, batch_size=1, shuffle=False)) \
        for name, dataset in zip(name_list, datasets)]
    )


    # Model container
    algos = []

    # Create the models
    for model_name, model_path in \
        zip(test_config['model_names'], test_config['model_path']):

        model_info = extract_model_name(model_name)

        #  model config
        model_config = set_model_parameters(model_info, config[model_info], base_config)
        model = set_model(model_info, model_config, device)

        # model setting (num_actions and initial_pulls (warm-up))
        model.set_name(model_name)
        model.set_device(device)
        model.to(device)

        #  check point
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint)
        algos.append(model)
    
    """
    if is_plot_training_curve:
        for model_name, train_result_path, val_result_path in \
            zip(test_config["model_names"], test_config["train_result"], test_config["val_result"]):

            plot_training_curve(train_result_path, val_result_path, \
                title= "training curve : {}".format(model_name),\
                result_path="./Result",
                file_name= "training_curve_{}.png".format(model_name)
            )

    """
    # Criterion
    criterion = elbo_loss_mse()
    
    # Trainer
    regressor_tester = meta_1d_regressor_trainer(algos, criterion, device, \
        list(dataloaders.values())[0],\
        savedir=save_result_dir,
        val_loaders=list(dataloaders.values())[1:],
        num_context=base_config['NUM_CONTEXT_POINTS'],\
        num_extra_target=dataloaders['rbf_periodic'].dataset._num_points - \
            base_config['NUM_CONTEXT_POINTS'],
        is_average_by_points=base_config['IS_AVERAGE_BY_POINTS']
    )

    # Test
    regressor_tester.test()
    


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
            "mean_context_ll", "std_context_ll", "mean_mse"]]
        np.savetxt(f, print_result, delimiter=',', fmt='%s')

        # Body
        for key, values in result.items():
            model_name = key
            mean_loss = np.asfarray(np.array(values[0]), float).mean()
            mean_ll = np.asfarray(np.array(values[1]), float).mean()
            std_loglikelihood = np.std(np.asfarray(np.array(values[1]), float))
            mean_context_ll = np.asfarray(np.array(values[3]), float).mean()
            std_context_ll = np.std(np.asfarray(np.array(values[3]), float))
            mean_mse = np.asfarray(np.array(values[4]), float).mean()

            print_result = [[key, mean_loss, mean_ll, std_loglikelihood,\
                mean_context_ll, std_context_ll, mean_mse]]

            np.savetxt(f, print_result, delimiter=',', fmt='%s')

        f.close()

if __name__ == "__main__":
    # test(is_plot_training_curve=False)
    
    
    # run test NPs
    for i in range(5):
        test()

    file_name_list = [
        "./Result/Result/result_test_rbf_n_freq_0_t_noise_0_is_random_0.txt",
        "./Result/Result/result_test_matern_n_freq_0_t_noise_0_is_random_0.txt",
        "./Result/Result/result_test_periodic_n_freq_0_t_noise_0_is_random_0.txt"
    ]
        
    # organize result statistics
    for file_name in file_name_list:
        # get result
        result = organize_result(file_name)
        
        # name
        file_path_split = os.path.split(file_name)

        # rename
        new_filename = "summary_" + str(file_path_split[1])
        new_filename = new_filename.split(".")[0] + ".csv"
        
        new_file_path = os.path.join("./", new_filename)
        
        # obtain summary
        get_statistic(result, new_file_path)
    





    



    




