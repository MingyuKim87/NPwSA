import os
import shutil 
import re

def make_dir(dirpath):
    try:
        if not(os.path.isdir(dirpath)):
            os.makedirs(os.path.join(dirpath))
    except OSError as e:
        if e.errno != errno.EEXIST:
            print("Failed to create directory!!!!!")
    
    return os.path.join(dirpath)

def organize_test_pt(path="/home/mgyukim/workspaces/AI701/save_models/sine/fixed_steps_200000_20210113", is_result_file=False):
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

                # Containers
                candidate_path_container = [0] * 3 if is_result_file else None
                candidate_file_name_container = [0]*3 if is_result_file else None
                txt_count = 1

                for file_name in file_list:
                    step = re.findall("\d+", file_name)
                    step = int(step[0]) if step else 0

                    if step >= last_step:
                        last_step = step
                        candidate_file_name = file_name
                        candidate_path = os.path.join(sub_sub_dir, file_name)
                        
                        if is_result_file:
                            candidate_path_container[0] = candidate_path 
                            candidate_file_name_container[0] = candidate_file_name 

                    if is_result_file:
                        if file_name.find("txt") > 0:
                            candidate_path = os.path.join(sub_sub_dir, file_name)
                            candidate_path_container[txt_count] = candidate_path
                            candidate_file_name_container[txt_count] = file_name
                            txt_count += 1

                    else:
                        continue

        # move the condidate model to 'test' dicrectory
        if is_result_file:
            for candidate_path, candidate_file_name in zip(candidate_path_container, candidate_file_name_container):
                shutil.copyfile(candidate_path, os.path.join(test_dir, component + "_" + candidate_file_name))
        else:
            shutil.copyfile(candidate_path, os.path.join(test_dir, component + "_" + candidate_file_name))

def move_save_models(model_path, destination):
    '''
        All cp models and remove all files except 'test' folder
    '''

    # copy all files to destination
    shutil.copytree(model_path, destination)

    # organize latest model file and copy to 'test' folder
    organize_test_pt(destination, True)

    # remove all directory except test folder
    model_list = os.listdir(destination)

    for model in model_list:
        model_dir_path = os.path.join(destination, model)

        sub_dir_list = os.listdir(model_dir_path)
        sub_dir_list.remove('test')

        # remove all directories
        for sub_dir in sub_dir_list:
            shutil.rmtree(os.path.join(model_dir_path, sub_dir))

    # Print
    print("*"*10, "Complete : Copy the latest model to test directories", "*"*10)

if __name__ == "__main__":
    cwd = os.getcwd()
    # 복사한 후 저장 위치
    destination = os.path.join(cwd, 'save_models', 'two_gp', 'fixed_steps_200000_decay_1e_3_20210324')
    
    # 복사할 파일 위치
    current_model_dir_path = "/home/mgyukim/workspaces/AI701/save_models/two_GP/fixed_steps_200000_decay_1e_3_20210324"
    move_save_models(current_model_dir_path, destination)
    
    


    
                    
                    



    