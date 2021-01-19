import numpy as np
import torch
import random
import os
import shutil
import torch.optim as optim
from torchvision.transforms import ToTensor, Resize, Compose
from torch.nn import functional as F
import torch.nn as nn

class HLoss(nn.Module):
    def __init__(self):
        super(HLoss, self).__init__()

    def forward(self, x):
        b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        b = -1.0 * b.sum(dim=1).mean()
        return b



def validate(net,loader,device):
    top1_val = AverageMeter('val_accuarcy')
    for step, batch in enumerate(loader):
        val_x, val_y_task = batch
        output = net(images=val_x.to(device))
        prec1 = accuracy(output, val_y_task.to(device), topk=(1,))[0]
        top1_val.update(prec1.item(), len(output))
    return top1_val.return_avg()


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res



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
        
    def initialize(self,net, train_loader):
        self.update_weighted_count()
        print('Saving representations to memory')
        # bar = Progbar(len(train_loader), stateful_metrics=[])
        for step, batch in enumerate(train_loader):
            with torch.no_grad():
                images = batch[0]
                index=batch[3]
                label=batch[2]

                images = images.to(self.device)

                output = net(images = images, mode = 0)
                self.weighted_sum[index, :] = output.cpu().numpy()
                self.memory[index, :] = self.weighted_sum[index, :]
                self.memory_label[index]=label.cpu().numpy()
                # bar.update(step, values= [])

    def update(self, index, values):
        self.weighted_sum[index, :] = values + (1 - self.weight) * self.weighted_sum[index, :]
        self.memory[index, :] = self.weighted_sum[index, :]/self.weighted_count
        pass

    def update_weighted_count(self):
        self.weighted_count = 1 + (1 - self.weight) * self.weighted_count

    def return_random(self, p, l):

        # allowed = [x for x in range(index[0])] + [x for x in range(index[0] + 1, 2000)]
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

    def save_model(self, model, current_value, test_acc,epoch,optimizer):
        if self.monitor_op(current_value, self.best):
            print('\nSave model, best val acc {:.3f}, epoch: {}'.format(current_value, epoch))
            self.best = current_value
            self.epoch_best = epoch
            self.test_best=test_acc
            checkpoint = {"model_state_dict": model.state_dict(),
                          "optimizer_state_dict": optimizer.state_dict(),
                          "epoch": epoch}
            torch.save(checkpoint , os.path.join(self.directory1,'best_model_epoch{}.pth'.format(epoch)))




class NoiseContrastiveEstimator():
    def __init__(self, device):
        self.device = device
    def __call__(self, temp,original_features, path_features, index, memory, negative_p = 0.1):
        loss = 0
        for i in range(original_features.shape[0]):

            cos = torch.nn.CosineSimilarity()
            criterion = torch.nn.CrossEntropyLoss()
            l=memory.memory_label[index[i]]
            negative=memory.return_random(p= negative_p, l=l)
            image_to_modification_similarity = cos(original_features[None, i,:], path_features[None, i,:])/temp
            matrix_of_similarity = cos(path_features[None, i,:],torch.Tensor(negative).to(self.device).detach() ) / temp


            similarities = torch.cat((image_to_modification_similarity, matrix_of_similarity))
            loss += criterion(similarities[None,:], torch.tensor([0]).to(self.device))
        return loss / original_features.shape[0]

