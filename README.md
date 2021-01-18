# Joint Contrastive Learning and Supervised Learning Towards  Domain Generalization
This is the pytorch implementation of the paper "Joint Contrastive Learning and Supervised Learning Towards  Domain Generalization".

## Requirements

* A Python install version 3.7
* A PyTorch and torchvision installation version 1.7.0 and 0.8.1, respectively.

## SETUP

*   The caffe model we used for [AlexNet](https://drive.google.com/file/d/1wUJTH1Joq2KAgrUDeKJghP1Wf7Q9w4z-/view?usp=sharing). Once downloaded, move it into ./alexnet_caffe.pth.tar
*  [PACS dataset](http://www.eecs.qmul.ac.uk/~dl307/project_iccv2017)
*  [OfficeHome dataset](https://www.hemanthdv.org/officeHomeDataset.html)
*  [CIFAR-10-C](https://github.com/hendrycks/robustness)

## Running experiments
### Multiple Domain Generalization **
You can train the model from scratch :
+ cd MDG
+ python main.py --data_dir your_data_dir --model AlexNet --datasets PACS
	+	data_dir: the dataset directory
	+	model: AlexNet or ResNet18
	+	datasets: PACS or OfficeHome

### Single Domain Generalization **
You can train the model from scratch :
+ cd SDG
+ python main.py --data_dir cifar10_dir --target_dir cifar10C_dir --file_name acc.csv
	+	data_dir: the dataset directory for cifar10
	+	target_dir: the dataset directory for cifar10-C
	+	file_name: the name of file storing the test acc
