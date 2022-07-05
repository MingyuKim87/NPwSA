import os
import shutil 
import re

import numpy as np

def make_dir(dirpath):
    try:
        if not(os.path.isdir(dirpath)):
            os.makedirs(os.path.join(dirpath))
    except OSError as e:
        if e.errno != errno.EEXIST:
            print("Failed to create directory!!!!!")
    
    return os.path.join(dirpath)

def organize_test_pt(path="/home/mgyukim/workspaces/AI701/save_models/sine/fixed_steps_200000_20210113"):
    # model_list
    model_list = os.listdir(path)

    # for all models
    for component in model_list:
        # sub_dir : model_name
        sub_dir = os.path.join(path, component)

        # make a test directory
        test_dir = make_dir(os.path.join(sub_dir, 'test'))

        if os.path.isdir(sub_dir):
            # sub_sub_dir : model_name/date
            sub_sub_dir_list = os.listdir(sub_dir)

            # remove 'test' folder in this list
            sub_sub_dir_list.remove('test')

            # lastest version (descending order)
            sub_sub_dir_name = sorted(sub_sub_dir_list, reverse=True)[0]

            # make path
            sub_sub_dir = os.path.join(sub_dir, sub_sub_dir_name)

            
            if os.path.isdir(sub_sub_dir):
                # file_list : model_name/date/model.pt
                file_list = os.listdir(sub_sub_dir)

                # last step
                last_step = 0

                for file_name in file_list:
                    step = re.findall("\d+", file_name)
                    step = int(step[0]) if step else 0

                    if step >= last_step:
                        last_step = step
                        candidate_file_name = file_name
                        candidate_path = os.path.join(sub_sub_dir, file_name)
                    else:
                        continue

        # move the condidate model to 'test' dicrectory
        shutil.copyfile(candidate_path, os.path.join(test_dir, component + "_" + candidate_file_name))

def move_save_models(model_path, destination):
    '''
        All cp models and remove all files except 'test' folder
    '''

    # copy all files to destination
    if os.path.isdir(destination):
        shutil.rmtree(os.path.join(destination))
    
    shutil.copytree(model_path, destination)

    model_list = os.listdir(destination)

    for model in model_list:
        model_dir_path = os.path.join(destination, model)

        sub_dir_list = os.listdir(model_dir_path)
        sub_dir_list = sorted(sub_dir_list, reverse=True)

        # define test directory
        test_dir = os.path.join(model_dir_path, 'test')
        
        # organize test directory
        if 'test' in sub_dir_list:
            sub_dir_list.remove('test')
            shutil.rmtree(test_dir)

        # remake test directories
        os.mkdir(os.path.join(model_dir_path, 'test'))
            
        # find latest model directories
        latest_model = os.path.join(model_dir_path, sub_dir_list[0])
        
        # list sub_directories
        saved_model_list = os.listdir(latest_model)

        # find last step .pts
        temp_number = 0
        selected_model_name = ""

    
        for model_name in saved_model_list:
            if 'txt' in model_name:
                continue
            
            #step_number = model_name.split("_")[-1]
            step_number_ = re.findall("\d+", model_name)[0]
            step_number = int(step_number_)

            if step_number >= temp_number:
                temp_number = step_number
                selected_model_name = model_name
                selected_model_file_path = os.path.join(latest_model, selected_model_name)

        # copy to test directory
        shutil.copy(selected_model_file_path, test_dir)

        # remove all directory except test folder
        for sub_dir in sub_dir_list:
            shutil.rmtree(os.path.join(model_dir_path, sub_dir))

    # Print
    print("*"*10, "Complete : Copy model's test directories", "*"*10)

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

def get_cwd_move_save_models(model_path, case):
    cwd = os.getcwd()

    if not case is str:
        NotImplementedError
    
    destination = os.path.join(cwd, 'save_models', case)
    
    model_path = "/home/mgyukim/workspaces/AI701/save_models/lv_model"
    move_save_models(model_path, destination)

if __name__ == "__main__":
    cwd = os.getcwd()
    destination = os.path.join(cwd, 'save_models', 'lv_models')
    
    model_path = "/home/mgyukim/workspaces/AI701/save_models/lv_model"
    move_save_models(model_path, destination)
    
    


    
                    
                    



    