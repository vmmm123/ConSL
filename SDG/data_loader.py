import os
import scipy.io as sio
import torch
import numpy as np
from torchvision import transforms
import torchvision
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import cv2


class GaussianBlur(object):
    # Implements Gaussian blur as described in the SimCLR paper
    def __init__(self, kernel_size, min=0.1, max=2.0):
        np.random.seed(0)
        self.min = min
        self.max = max
        # kernel size is set to be 10% of the image height/width
        self.kernel_size = kernel_size

    def __call__(self, sample):
        sample = np.array(sample)

        # blur the image with a 50% chance
        prob = np.random.random_sample()

        if prob < 0.5:
            sigma = (self.max - self.min) * np.random.random_sample() + self.min
            sample = cv2.GaussianBlur(sample, (self.kernel_size, self.kernel_size), sigma)

        return sample

def loader(args):
    source_dataset = Loader_unif_sampling_dataset(args)

    train_loader = torch.utils.data.DataLoader(dataset=source_dataset, batch_size=args.batch_size, shuffle=True,
                                               num_workers=16)

    normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
    img_transform_test = transforms.Compose([transforms.ToTensor(), normalize])

    valid_dataset = torchvision.datasets.CIFAR10(root=args.data_dir, train=False, transform=img_transform_test)
    valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size=300, num_workers=32)



    lables = np.load(os.path.join(args.target_data_dir,'labels.npy'))
    lables = torch.from_numpy(lables[-10000:])
    target_loaders = []
    for method_name in args.target:
        cifar_c = np.load(os.path.join(args.target_data_dir,method_name + '.npy'))
        cifar_c = cifar_c[-10000:]
        target_dataset = GetLoader(cifar_c, lables, transform=img_transform_test)
        target_loaders.append(DataLoader(target_dataset, batch_size=300, num_workers=16))

    return train_loader,valid_loader,target_loaders


class GetLoader(Dataset):

    def __init__(self, data_root, data_label,transform):
        self.data = data_root
        self.label = data_label
        self.transform=transform
    def __getitem__(self, index):
        data = self.data[index]
        labels = self.label[index]
        return self.transform(data), labels

    def __len__(self):
        return len(self.data)


class Loader_unif_sampling_dataset(Dataset):
    def __init__(self, args):

        self.dataset = torchvision.datasets.CIFAR10(root=args.data_dir, train=True)
        self.length  =50000


        normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                         std=[x / 255.0 for x in [63.0, 62.1, 66.7]])

        color_jitter = transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)

        self.trans_t = transforms.Compose(
            [transforms.ToTensor(),
        	transforms.Lambda(lambda x: F.pad(x.unsqueeze(0),
        						(4,4,4,4),mode='reflect').squeeze()),
            transforms.ToPILImage(),transforms.RandomResizedCrop(32 ), transforms.RandomHorizontalFlip(),
             transforms.RandomApply([color_jitter], p=0.8), transforms.RandomGrayscale(p=0.2),
             GaussianBlur(kernel_size=5), transforms.ToTensor(),
             normalize]
        )


    def __getitem__(self, idx):
        data_1, y_task_1 = self.dataset.__getitem__(idx)
        jigsaw_1 = self.trans_t(data_1)
        data_1 = self.trans_t(data_1)

        return data_1, jigsaw_1, torch.tensor(y_task_1).long().squeeze() ,idx

    def __len__(self):
        return self.length

