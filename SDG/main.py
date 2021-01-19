import torch
import torch.nn as nn
import numpy as np
from torch.optim.lr_scheduler import StepLR
from data_loader import *
from torchvision import transforms
from util import *
from wideresnet import WideResNet
import argparse




def main(args):

    print('Using CIFAR-10 data')
    checkpoint = ModelCheckpoint(mode='max', directory1=args.save_dir)
    train_loader,valid_loader,target_loaders=loader(args)

    net= WideResNet(depth=args.depth, num_classes=args.num_classes,widen_factor=args.widen_factor).to(args.device)
    net.train()
    optimizer = torch.optim.SGD(net.parameters(), args.lr,
                                    momentum=args.momentum, nesterov = True,
                                    weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(train_loader)*args.epochs)

    memory = Memory(size = 50000, weight= 0.5, device = args.device,num_features=args.num_reduced_festures)
    memory.initialize(net, train_loader)

    with open(os.path.join(args.save_dir,args.file_name), 'w') as file:
        file.write(
            'epoch,')
        for method_name in args.target:
            file.write(
                '{},'.format(method_name))
        file.write('avg\n')


    for epoch in range(0,args.epochs):

        print('\nEpoch: {}'.format(epoch))
        memory.update_weighted_count()
        net.train()
        net=train(train_loader,net,optimizer,scheduler,memory,args)

        net.eval()
        val_acc=validate(net, valid_loader,args.device)

        acc=[]
        with open(os.path.join(args.save_dir,args.file_name), 'a') as file:
            file.write(
                '{},'.format(epoch))
            for target_loader in target_loaders:
                test_acc=validate(net, target_loader,args.device)
                file.write(
                    '{},'.format(test_acc))
                acc.append(test_acc)
            test_acc = np.mean(acc)
            file.write(
                '{}\n'.format(test_acc))


        # save model if improved
        print("val_acc:{:.4f}, test_acc_avg:{:.4f}".format(val_acc,test_acc))
        checkpoint.save_model(net, val_acc, test_acc, epoch,optimizer)
    print("Test accuarcy:{:.4f} by the best model at epoch {}\n".format(checkpoint.test_best,
                                                                               checkpoint.epoch_best))


def train(train_loader,net,optimizer,scheduler,memory,args):
    noise_contrastive_estimator = NoiseContrastiveEstimator(args.device)
    class_criterion = nn.CrossEntropyLoss()
    entropy_criterion = HLoss()
    train_loss = AverageMeter('train_loss')
    cls_loss = AverageMeter('cls_loss')
    loss_image = AverageMeter('loss_image')
    loss_image2 = AverageMeter('loss_image2')
    loss_entropy = AverageMeter('loss_entropy ')

    for step, batch in enumerate(train_loader):
        images, images2, y_task, index = batch

        # prepare batch
        images = images.to(args.device, non_blocking=True)
        images2 = images2.to(args.device, non_blocking=True)
        y_task = y_task.to(args.device, non_blocking=True)

        optimizer.zero_grad()
        # forward, loss, backward, step
        output = net(images=images, images2=images2, mode=1)

        representations = memory.return_representations(index).to(args.device, non_blocking=True).detach()
        loss_1 = noise_contrastive_estimator(args.temperature, representations, output[1],
                                             index,
                                             memory, negative_p=args.negative_p,
                                             )
        loss_2 = noise_contrastive_estimator(args.temperature, representations, output[0],
                                             index,
                                             memory,  negative_p=args.negative_p)
        loss_cls = class_criterion(output[2], y_task)

        loss_ent = entropy_criterion(output[2])
        loss = args.alpha *(loss_1 + loss_2) + loss_cls + loss_ent

        loss.backward()

        optimizer.step()
        scheduler.step()
        # update representation memory
        memory.update(index, output[0].detach().cpu().numpy())

        # update metric
        train_loss.update(loss.item(), images.shape[0])
        loss_image.update(loss_1.item(), images.shape[0])
        loss_image2.update(loss_2.item(), images.shape[0])
        loss_entropy.update(loss_ent.item(), images.shape[0])
        cls_loss.update(loss_cls.item(), images.shape[0])

    print("train_loss:{:.4f}\t".format(train_loss.return_avg()))
    return net

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='../../../data/xiemengwei',help='data folder')
    parser.add_argument('--target_data_dir', default='../../../data/xiemengwei/CIFAR-10-C/', help='target domains folder')
    parser.add_argument('--save_dir', default='save',help='folder to save model and results')
    parser.add_argument('--file_name', default='test_acc.csv', help='file to save test results per epoch')
    parser.add_argument('--gpu', type=int, default=4,help='GPU id')
    parser.add_argument('--depth', type=int,default=16, help='depth of wideresnet')
    parser.add_argument('--widen_factor', type=int, default=4, help='widen_factor of wideresnet')
    parser.add_argument('--num_classes', type=int, default=10, help='number of categories')
    parser.add_argument('--negative_p', type=float, default=0.1, help='proportion of negative examples in NCE')
    parser.add_argument('--temperature', type=float, default=0.07, help='temperature in NCE')
    parser.add_argument('--num-reduced-festures', type=int, default=256, help='dimension number of projected features')
    parser.add_argument('--lr', type=float, default=0.1, help='initial learning rate')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size for each source domain')
    parser.add_argument('--epochs', type=int, default=200, help='number of training epoches')
    parser.add_argument('--alpha', type=float, default=0.5, help='the weight of self_supeivisor')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')



    args = parser.parse_args()
    args.device = torch.device("cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu")
    args.target= ['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur',
         'snow', 'frost', 'fog','brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression', 'speckle_noise',
         'gaussian_blur', 'spatter', 'saturate']


    main(args)