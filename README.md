# Style-Invariant Domain Generalization Via Contrastive Learning
This is the pytorch implementation of the paper "Style-Invariant Domain Generalization Via Contrastive Learning".

## Requirements

* A Python install version 3.7
* A PyTorch and torchvision installation version 1.7.0 and 0.8.1, respectively.

## Running experiments
* This repo uses PACS on AlexNet as an illustrative example:   
    *   The caffe model we used for [AlexNet](https://drive.google.com/file/d/1wUJTH1Joq2KAgrUDeKJghP1Wf7Q9w4z-/view?usp=sharing). Once downloaded, move it into ./alexnet_caffe.pth.tar
    *  [PACS dataset](http://www.eecs.qmul.ac.uk/~dl307/project_iccv2017)
* You can train the model from scratch 
    * python main.py --data_dir your_data_dir --checkpoint_dir your_save_root  --target sketch 
    * your_data_dir: the dataset directory
    * your_save_root: the directory for saving the models
    * target: the  target domain while the rest domians are used for training
    
                                              
Note：We make a mistake in the paper that α(alpha) should be 0.5 rather than 1 to obtain the best results.
