import numpy as np
import torch
import random
import os
import torch.optim as optim
from torch import nn
from torch.nn import functional as F
import cv2
import numpy as np
from model import *



def get_model(model):
    if model=='AlexNet':
        return AlexNet
    elif model=='ResNet18':
        return ResNet18
    else:
        raise ValueError('Name of network unknown %s' % args.model)


def get_lr(net,args):
    if args.model=='AlexNet':
        return [(net.features, 1.0), (net.classifier, 1.0),
                        (net.class_classifier, 1.0 * args.fc_weight), (net.discriminator, 1.0 * args.disc_weight),
                           (net.projection_original_features, 1.0 * args.proj_weight)]
    elif args.model=='ResNet18':
        return [(net.model, 1.0),
                (net.class_classifier, 1.0 * args.fc_weight), (net.domain_classifier, 1.0 * args.disc_weight),
                   (net.projection_original_features, 1.0 * args.proj_weight)]


def get_optimizer(model, init_lr, momentum, weight_decay,  nesterov=True):

    params_to_update = model.parameters()
    optimizer = optim.SGD(
        params_to_update, lr=init_lr, momentum=momentum,
        weight_decay=weight_decay, nesterov=nesterov)

    return optimizer

def validate(test_loader,net,device):
    total = 0
    correct = 0

    for step, batch in enumerate(test_loader):
        val_x, val_y_task = batch
        _, predicted = torch.max(net(images=val_x.to(device),mode=2), 1)
        total += val_y_task.size(0)
        correct += (predicted.cpu() == val_y_task).sum().item()
    val_acc = correct / total
    return val_acc


class HLoss(nn.Module):
    def __init__(self):
        super(HLoss, self).__init__()

    def forward(self, x):
        b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        b = -1.0 * b.sum(dim=1).mean()
        return b





class AverageMeter(object):
    '''
    Taken from:
    https://github.com/keras-team/keras
    '''
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
    def return_avg(self):
        return self.avg



class Memory(object):
    def __init__(self, device, size = 2000, weight = 0.5,num_features=128):
        self.memory = np.zeros((size, num_features))
        self.memory_label = np.zeros(size)
        self.weighted_sum = np.zeros((size, num_features))
        self.weighted_count = 0
        self.weight = weight
        self.device = device
        self.num_features=num_features
        
    def initialize(self,i, net, train_loader):
        self.update_weighted_count()
        print('Saving representations to memory %d'%i)
        for step, batch in enumerate(train_loader):
            with torch.no_grad():
                images = batch[i]
                index=batch[i+3]
                label=batch[i+9]

                images = images.to(self.device)
                _,output = net(images = images, mode = 0)
                self.weighted_sum[index, :] = output.cpu().numpy()
                self.memory[index, :] = self.weighted_sum[index, :]
                self.memory_label[index]=label.cpu().numpy()


    def update(self, index, values):
        self.weighted_sum[index, :] = values + (1 - self.weight) * self.weighted_sum[index, :]
        self.memory[index, :] = self.weighted_sum[index, :]/self.weighted_count
        pass

    def update_weighted_count(self):
        self.weighted_count = 1 + (1 - self.weight) * self.weighted_count

    def return_random(self, p, l):
        allowed=[]
        for i,label in enumerate(self.memory_label):
            if label!=l:
                allowed.append(i)
        size=int(len(allowed)*p)
        index = random.sample(allowed, size)
        return self.memory[index,:]

    def return_representations(self, index):
        if isinstance(index, torch.Tensor):
            index = index.tolist()
        return torch.Tensor(self.memory[index,:])



class ModelCheckpoint():
    def __init__(self, mode, directory1):
        self.directory1 = directory1
        self.epoch_best=0
        self.best = 0
        self.test_best=0
        if not os.path.isdir(self.directory1):
            os.mkdir(self.directory1)

        if mode =='min':
            self.best = np.inf
            self.monitor_op = np.less
        elif mode == 'max':
            self.best = 0
            self.monitor_op = np.greater
        else:
            print('\nChose mode \'min\' or \'max\'')
            raise Exception('Mode should be either min or max')

    def save_model(self, model, current_value, test_acc,epoch,target):
        if self.monitor_op(current_value, self.best):
            print('\nSave model, best val acc {:.4f}, epoch: {}'.format(current_value, epoch))
            self.best = current_value
            self.epoch_best = epoch
            self.test_best=test_acc

            torch.save(model.state_dict(), os.path.join(self.directory1,'{}_best_epoch.pth'.format(target)))



class NoiseContrastiveEstimator():
    def __init__(self, device):
        self.device = device
    def __call__(self, d_i,temp,original_features, path_features, index, memorys, num_features=128,negative_p = 0.1,num_domains=3):
        loss = 0
        for i in range(original_features.shape[0]):

            cos = torch.nn.CosineSimilarity()
            criterion = torch.nn.CrossEntropyLoss()
            negative=[]
            l=memorys[d_i].memory_label[index[i]]
            for j in range(num_domains):
                n=memorys[j].return_random(p= negative_p, l=l)
                negative.append(torch.Tensor(n).to(self.device).detach())
            negative = torch.cat(negative, dim=0)
            image_to_modification_similarity = cos(original_features[None, i,:], path_features[None, i,:])/temp
            matrix_of_similarity = cos(path_features[None, i,:], negative) / temp
            similarities = torch.cat((image_to_modification_similarity, matrix_of_similarity))
            loss += criterion(similarities[None,:], torch.tensor([0]).to(self.device))
        return loss / original_features.shape[0]


class GaussianBlur(object):
    # Implements Gaussian blur as described in the SimCLR paper
    def __init__(self, kernel_size, min=0.1, max=2.0):
        self.min = min
        self.max = max
        # kernel size is set to be 10% of the image height/width
        self.kernel_size = kernel_size
        np.random.seed(0)

    def __call__(self, sample):
        sample = np.array(sample)

        # blur the image with a 50% chance
        prob = np.random.random_sample()

        if prob < 0.5:
            sigma = (self.max - self.min) * np.random.random_sample() + self.min
            sample = cv2.GaussianBlur(sample, (self.kernel_size, self.kernel_size), sigma)

        return sample