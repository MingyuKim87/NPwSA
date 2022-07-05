######### Description of neural process with stochastic attention for lotka volterra processes #########

1. Run training 
python main.py --device=[device_number] --model=[model_name]

e.g) python main.py --device=0 --model=ANP_weibull_300

    - model_name    
        CNP : conditional neural process
        NP : neural process
        ANP : attentive neural process
        ANP_variational : ANP with infomation dropout
        ANP_weibull_300 : The proposed method


2. Dataset adding noise

    In "main.py", we can manipulate option for dataset with noises or not. (line 63)
        if "True", the training set is the Lotka volterra + periodic noise
        else ("False"), the training set is only the Lotka volterra process

    By default, this options is set as "False".
    

3. Location of checkpoints
    /save_models/[model_name]/[current_date]


4. Run testing (Hudson's hare_lynx data)

    python main.py --is_hare_lynx=True

    You can select a model's checkpoint on (./config/test.yml)






