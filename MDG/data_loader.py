from PIL import Image
import os
import torch
import numpy as np
from torchvision import datasets,transforms
import random
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader,ConcatDataset
from util import *


def loader(data_dir,source,target,batch_size,num_domains=3):
    img_transform_test = transforms.Compose([transforms.Resize((225, 225)), transforms.ToTensor(),
                                             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    training_dataset = []
    validation_dataset = []

    target_path = data_dir + target
    target_dataset = Loader_validation(path=target_path, transform=img_transform_test)
    target_loader = torch.utils.data.DataLoader(dataset=target_dataset, batch_size=50, num_workers=16)

    for i in range(num_domains):
        splitter = ImageFolderSplitter(data_dir + source[i], class_to_idx=target_dataset.class_to_idx())
        x_train, y_train = splitter.getTrainingDataset()
        training_dataset.append(DatasetFromFilename(x_train, y_train))

        x_valid, y_valid = splitter.getValidationDataset()
        validation_dataset.append(DatasetFromFilename(x_valid, y_valid, transforms=img_transform_test))

    source_dataset = Loader_unif_sampling_dataset(training_dataset[0], training_dataset[1], training_dataset[2])
    train_loader = DataLoader(dataset=source_dataset, batch_size=batch_size, shuffle=True, num_workers=16)

    valid_dataset = ConcatDataset([validation_dataset[0], validation_dataset[1], validation_dataset[2]])
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=200, num_workers=32)
    return train_loader,valid_loader,target_loader,source_dataset.len


class ImageFolderSplitter:
    # images should be placed in folders like:
    # --root
    # ----root\dogs
    # ----root\dogs\image1.png
    # ----root\dogs\image2.png
    # ----root\cats
    # ----root\cats\image1.png
    # ----root\cats\image2.png
    # path: the root of the image folder
    def __init__(self, path, class_to_idx, train_size=0.9):
        self.path = path
        self.train_size = train_size
        self.class2num = class_to_idx
        self.num2class = {}
        self.class_nums = {}
        self.data_x_path = []
        self.data_y_label = []
        self.x_train = []
        self.x_valid = []
        self.y_train = []
        self.y_valid = []
        for root, dirs, files in os.walk(path):

            if len(files) > 1 and len(dirs) == 0:
                category = ""
                for key in self.class2num.keys():
                    if key in root:
                        category = key
                        break
                label = self.class2num[category]
                self.class_nums[label] = 0
                for file1 in files:
                    self.data_x_path.append(os.path.join(root, file1))
                    self.data_y_label.append(label)
                    self.class_nums[label] += 1

        self.x_train, self.x_valid, self.y_train, self.y_valid = train_test_split(self.data_x_path, self.data_y_label,
                                                                                  shuffle=True,
                                                                                  train_size=self.train_size)

    def getTrainingDataset(self):
        return self.x_train, self.y_train

    def getValidationDataset(self):
        return self.x_valid, self.y_valid


class Loader_unif_sampling_dataset(Dataset):
    def __init__(self, dataset_1, dataset_2, dataset_3):
        self.dataset_1 = dataset_1
        self.dataset_2 = dataset_2
        self.dataset_3 = dataset_3

        self.len = [self.dataset_1.__len__(), self.dataset_2.__len__(), self.dataset_3.__len__()]
        self.length = max(self.len)

        color_jitter = transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)
        self.trans_t = transforms.Compose(
            [transforms.RandomResizedCrop(225, scale=(0.7, 1.0)), transforms.RandomHorizontalFlip(),
             transforms.RandomApply([color_jitter], p=0.8), transforms.RandomGrayscale(p=0.2),
             GaussianBlur(kernel_size=21), transforms.ToTensor(),
             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
        )


    def __getitem__(self, idx):
        if idx/self.len[0]<1:
            idx_1 = idx % self.len[0]
        else:
            idx_1=random.randint(0,self.len[0]-1)
        if idx/self.len[1]<1:
            idx_2 = idx % self.len[1]
        else:
            idx_2=random.randint(0,self.len[1]-1)
        if idx/self.len[2]<1:
            idx_3 = idx % self.len[2]
        else:
            idx_3=random.randint(0,self.len[2]-1)

        data_1, y_task_1 = self.dataset_1.__getitem__(idx_1)
        y_domain_1 = 0.
        other_1 = self.trans_t(data_1)
        data_1 = self.trans_t(data_1)

        data_2, y_task_2 = self.dataset_2.__getitem__(idx_2)
        y_domain_2 = 1.
        other_2 = self.trans_t(data_2)
        data_2 = self.trans_t(data_2)

        data_3, y_task_3 = self.dataset_3.__getitem__(idx_3)
        y_domain_3 = 2.
        other_3 = self.trans_t(data_3)
        data_3 = self.trans_t(data_3)

        return data_1, data_2, data_3, idx_1, idx_2, idx_3, other_1, other_2, other_3, torch.tensor(
            y_task_1).long().squeeze(), torch.tensor(y_task_2).long().squeeze(), torch.tensor(
            y_task_3).long().squeeze(), torch.tensor(y_domain_1).long().squeeze(), torch.tensor(
            y_domain_2).long().squeeze(), torch.tensor(y_domain_3).long().squeeze()

    def __len__(self):
        return self.length



class DatasetFromFilename(Dataset):
    # x: a list of image file full path
    # y: a list of image categories
    def __init__(self, x, y, transforms=None):
        super(DatasetFromFilename, self).__init__()
        self.x = x
        self.y = y
        self.transform = transforms

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        img = Image.open(self.x[idx])
        img = img.convert("RGB")
        if self.transform == None:
            return img, self.y[idx]
        else:
            return self.transform(img), self.y[idx]


class Loader_validation(Dataset):
    def __init__(self, path, transform=None):
        self.path = path
        self.dataset = datasets.ImageFolder(path, transform=transform)
        self.length = self.dataset.__len__()

    def class_to_idx(self):
        return self.dataset.class_to_idx

    def __getitem__(self, idx):
        data, y_task = self.dataset.__getitem__(idx)

        return data, torch.tensor(y_task).long().squeeze()

    def __len__(self):
        return self.length


