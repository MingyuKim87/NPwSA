######### Description of neural process for movielenz10k dataset #########

1. Run training 
    * Prior to running training, you should make comments from lines 306 to 315 in "main.py"

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

    
3. Run testing (Hudson's hare_lynx data)
    * Prior to running test, you should make comments line 314, 315 in "main.py"
    
    python main.py --is_hare_lynx=True

    You can select a model's checkpoint on the file (./config/test.yml).






