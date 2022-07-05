######### Description of neural process for celebA dataset #########

1. You should identify location of the root directory of the celebA dataset.
    e.g) /home/[Username]/Data/celebA

2. You should describe this location on the line 50 and 123 in "./main.py"

3. Run training 
    * Prior to running test, you should make comments line 336 in "main.py"
python main.py --device=[device_number] --model=[model_name]

e.g) python main.py --device=0 --model=ANP_weibull_300

    - model_name    
        CNP : conditional neural process
        NP : neural process
        ANP : attentive neural process
        ANP_variational : ANP with infomation dropout
        ANP_weibull_100 : The proposed method

4. Location of checkpoints
    /save_models/[model_name]/[current_date]


5. Select test dataset 
    In ./config/models.yml

    There is option "ATTR_TEST" to choose the class of celebA.
    In this experiment, we have three options; "Bald", "Black_Hair" or None
        * None means we use all celebA images.  
    
6. Run testing
    * You can select a model's checkpoint on the file (./config/test.yml).
    
    * Prior to running test, you should define the number of context point array.
        e.g) test(is_several_contexts=True, num_context_list=[50, 100, 150, 200, 300, 400, 500, 600])
    
    * run this command.
        python main.py

    






