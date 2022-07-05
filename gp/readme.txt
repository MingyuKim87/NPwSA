######### Description of neural process with stochastic attention for 1D regression #########

1. Run training 
python main.py --device=[device_number] --model=[model_name]

e.g) python main.py --device=0 --model=ANP_weibull_300

    - model_name    
        CNP : conditional neural process
        NP : neural process
        ANP : attentive neural process
        ANP_variational : ANP with infomation dropout
        ANP_weibull_300 : The proposed method

2. Location of checkpoints
    /save_models/[model_name]/[current_date]

3. Select the type of datasets
In main.py, there are three types of dataset.
The first is the RBF GP with random parameters (line : 58)
The second is the RBF GP with fixed parameters (line : 67)
By default, the dataset is the RBF GP with fixed parameters adding periodic noises. (line : 77)

