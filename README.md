# Joint Contrastive Learning and Supervised Learning Towards  Domain Generalization
This is the pytorch implementation of the paper "Joint Contrastive Learning and Supervised Learning Towards  Domain Generalization".

## Requirements

* A Python install version 3.7
* A PyTorch and torchvision installation version 1.7.0 and 0.8.1, respectively.

## SETUP
* This repo contains codes for PACS on AlexNet :   
    *   The caffe model we used for [AlexNet](https://drive.google.com/file/d/1wUJTH1Joq2KAgrUDeKJghP1Wf7Q9w4z-/view?usp=sharing). Once downloaded, move it into ./alexnet_caffe.pth.tar
    *  [PACS dataset](http://www.eecs.qmul.ac.uk/~dl307/project_iccv2017)
    *  [OfficeHome dataset](https://www.hemanthdv.org/officeHomeDataset.html)
    
## Running experiments
* You can train the model from scratch 
    * python main.py --data_dir your_data_dir --checkpoint_dir your_save_root  --datasets PACS
    
    * your_data_dir: the dataset directory
    * your_save_root: the directory for saving the models
    * model：AlexNet or ResNet18
    * datasets： PACS or OfficeHome
    

