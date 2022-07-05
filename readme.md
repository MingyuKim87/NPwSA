Copyright (c) 2020-2022 the authors : Neural Processes with Stochastic Attention

## Description of experiments : Neural Processes with Stochastic Attention  ##
</hr>

### There are 4 experiments in this code.
* gp / gp_test : 1D regression task
* recommend_sys : movieLenz-10k
* lv_model : predator-prey model
* celebA : image completion task

### The structure of directory
* recommend_sys / lv_model / celebA include train, test codes 
        "./save_models/" : location of model checkpoints in training
        "./test_save_models/" : model path in testing

* In gp / gp_test, we separate train, test codes.
        gp : train code
            "./save_models/" : location of model checkpoints in training
        gp_test : test code
            "./save_models/" : model path in testing

### Dataset
* gp : it is synthetic dataset, so it dose not explicitly requires the data file. 
* recommend_sys : "./data/movielens"
* lv_model : only test dataset (hudson's hare lynx) you can assess to "./data/dataset/LynxHare.txt"
         For the training dataset, you should refer "https://github.com/juho-lee/bnp".
            . you have to run "/regression/data/lotka_volterra.py" and then obtain "train.tar" / "val.tar" 
            . you must place "train.tar" and "val.tar" on "./lv_model/data/dataset/"
    - celebA : you can download at "https://www.kaggle.com/jessicali9530/celeba-dataset"
            . you should write the absolute root path of celebA dataset on the line 50 and 123 in "./celebA/main.py"

### Run 
    - At each directory, there is "readme.txt", which describe how to run train and test codes.

### Pre-trained checkpoints
    - gp_test / lv_model / recommend_sys : there exist pre-trained checkpoints. 
    - celebA : There are no pre-trained checkpoint due to space constraints. 


