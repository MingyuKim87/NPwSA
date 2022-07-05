######### Description of Testing trained models on the RBF GP, the Matern32 GP and the periodic GP #########

1. Run testing
    By default, we can test all models trained on the RBF GP with fixed parameters adding periodic noises. 

    python main.py 


2. Config

    ./config/test.yml

    We provide of pre-trained models for the RBF GP 1D regression.
    There are two types of models
        1. the RBF GP with fixed parameters (./save_models/rbf_gp_fixed/)
        2. the RBF GP with fixed parameters adding periodic noises (./save_models/rbf_gp_fixed_adding_noises/)

    
3. Result files

    By default, we run 5 times and calculate the average likelihood and its standard deviation.
    All result files are automatically saved in the root directory

    e.g) ./*.csv 


    



