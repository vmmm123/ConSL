import torchvision
import torch
import torch.nn as nn
import numpy as np
from torch.optim.lr_scheduler import StepLR
from data_loader import *
from torchvision import transforms
from model import *
from util import *
import argparse
from torch.utils import model_zoo


def main(args):
    device = torch.device("cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu")

    if args.datasets=='PACS':
        args.domains=['photo', 'cartoon', 'sketch', 'art_painting']
    elif args.datasets=='OfficeHome':
        args.domains = ['Product','Clipart','Art','RealWorld']
    for arg in vars(args):
        print (arg, getattr(args, arg))


    for target in args.domains:
        source=args.domains[:]
        source.remove(target)
        print("Target domain: {}, Training domans: {}, {}, {}\n".format(target, source[0], source[1], source[2]))
        checkpoint = ModelCheckpoint(mode='max', directory1=args.checkpoint_dir)
        train_loader, valid_loader, target_loader,datasets_len=loader(args.data_dir,source,target,batch_size=args.batch_size,num_domains=args.num_domains)
        net = get_model(args.model)(num_classes=args.num_classes, num_features=args.num_reduced_festures).to(device)
        model_lr = get_lr(net, args)

        optimizers = [get_optimizer(model_part, args.lr * i, args.momentum, args.weight_decay) for model_part, i in model_lr]
        schedulers = [StepLR(optimizer=opt, step_size=args.lr_step, gamma=args.lr_decay_gamma)
                             for opt in optimizers]

        memorys = []
        for i in range(args.num_domains):
            memorys.append(Memory(size = datasets_len[i], weight= 0.5, device = device,num_features=args.num_reduced_festures))
            memorys[i].initialize(i, net, train_loader)


        if args.model=='AlexNet':
            state_dict = torch.load("../alexnet_caffe.pth.tar")
            del state_dict["classifier.fc8.weight"]
            del state_dict["classifier.fc8.bias"]
            net.features.load_state_dict(state_dict, strict=False)
            net.classifier.load_state_dict(state_dict, strict=False)
        elif args.model=='ResNet18':
            net.model.load_state_dict(model_zoo.load_url(model_urls['resnet18']), strict=False)

        for epoch in range(0,args.epochs):

            print('\nEpoch: {}'.format(epoch))
            net.train()
            net=train(train_loader, net,optimizers,schedulers,memorys,args,device)

            net.eval()
            val_acc = validate(valid_loader,net,device)
            # save model if improved

            test_acc=validate(target_loader,net,device)

            print("val_acc:{:.4f}\ttest_acc:{:.4f}\n".format(val_acc,test_acc))
            checkpoint.save_model(net, val_acc, test_acc, epoch,  target)

        print("Test accuarcy:{:.4f} on {} by the best model at epoch {}:\n".format(checkpoint.test_best,target,checkpoint.epoch_best))


def train(train_loader, net,optimizers ,schedulers,memorys,args,device):
    for memory in memorys:
        memory.update_weighted_count()
    train_loss = AverageMeter('train_loss')
    cls_loss = AverageMeter('cls_loss')
    domain_loss = AverageMeter('domain_loss')
    loss_image = AverageMeter('loss_image')
    loss_image2 = AverageMeter('loss_image2')
    loss_entropy = AverageMeter('loss_entropy ')
    noise_contrastive_estimator = NoiseContrastiveEstimator(device)
    class_criterion = nn.CrossEntropyLoss().to(device)
    entropy_criterion = HLoss().to(device)


    for step, batch in enumerate(train_loader):
        x_1, x_2, x_3, idx_1, idx_2, idx_3, o_x_1, o_x_2, o_x_3, y_task_1, y_task_2, y_task_3, y_domain_1, y_domain_2, y_domain_3 = batch
        length = len(idx_1)
        images = torch.cat((x_1, x_2, x_3), dim=0)
        y_task = torch.cat((y_task_1, y_task_2, y_task_3), dim=0)
        y_domain = torch.cat((y_domain_1, y_domain_2, y_domain_3), dim=0)
        index = [idx_1, idx_2, idx_3]
        images2 = torch.cat((o_x_1, o_x_2, o_x_3), dim=0)

        # prepare batch
        images = images.to(device)
        images2 = images2.to(device)
        y_task = y_task.to(device)
        y_domain = y_domain.to(device)
        for optimizer in optimizers:
            optimizer.zero_grad()

        # loss backward
        output = net(images=images, images2=images2, mode=1)

        loss_1 = 0
        loss_2 = 0
        for i in range(args.num_domains):
            representations = memorys[i].return_representations(index[i]).to(device).detach()
            loss_1 += noise_contrastive_estimator(i, args.temperature, representations, output[1][i * length:(i + 1) * length],
                                                  index[i],
                                                  memorys, num_features=args.num_reduced_festures, negative_p=args.negative_p,
                                                  num_domains=args.num_domains)
            loss_2 += noise_contrastive_estimator(i, args.temperature, representations, output[0][i * length:(i + 1) * length],
                                                  index[i],
                                                  memorys, num_features=args.num_reduced_festures, negative_p=args.negative_p,
                                                  num_domains=args.num_domains)
        loss_cls = class_criterion(output[2], y_task)
        loss_ent = entropy_criterion(output[2])
        loss_domain = class_criterion(output[3], y_domain)
        loss = args.alpha * (loss_1 + loss_2) + args.beta * loss_domain + loss_cls + loss_ent

        loss.backward()

        for optimizer in optimizers:
            optimizer.step()

        # update representation memory
        for i in range(args.num_domains):
            memorys[i].update(index[i], output[0][i * length:(i + 1) * length].detach().cpu().numpy())

        # update metric and bar
        train_loss.update(loss.item(), images.shape[0])
        loss_image.update(loss_1.item(), images.shape[0])
        loss_image2.update(loss_2.item(), images.shape[0])
        loss_entropy.update(loss_ent.item(), images.shape[0])
        cls_loss.update(loss_cls.item(), images.shape[0])
        domain_loss.update(loss_domain.item(), images.shape[0])

    for scheduler in schedulers:
        scheduler.step()
    print("train_loss:{:.4f}\t".format(train_loss.return_avg()))
    return net


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='../../../data/xiemengwei/OfficeHomeDataset/',help='data folder')
    parser.add_argument('--checkpoint_dir', default='save',help='folder to save model')
    parser.add_argument('--datasets', default='OfficeHome', help='the datasets for training and testing', choices=['PACS', 'OfficeHome'])
    parser.add_argument('--gpu', type=int, default=3,help='GPU id')
    parser.add_argument('--model', default='ResNet18', help='basic model', choices=['AlexNet', 'ResNet18'])
    parser.add_argument('--num-domains', type=int, default=3, help='number of source domians')
    parser.add_argument('--num_classes', type=int, default=65, help='number of categories')
    parser.add_argument('--negative_p', type=float, default=0.1, help='proportion of negative examples in NCE')
    parser.add_argument('--temperature', type=float, default=0.07, help='temperature in NCE')
    parser.add_argument('--num-reduced-festures', type=int, default=256, help='dimension number of projected features')
    parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size for each source domain')
    parser.add_argument('--epochs', type=int, default=30, help='number of training epoches')
    parser.add_argument('--alpha', type=float, default=0.5, help='the weight of self_supeivisor')
    parser.add_argument('--beta', type=float, default=0.5, help='the weight of domain adv')
    parser.add_argument('--lr_step', type=int, default=10, help='number of steps the learning rate decays')
    parser.add_argument('--lr_decay-gamma', type=float, default=0.2, help='ratio of learning rate decay')
    parser.add_argument('--proj_weight', type=float, default=10.0, help='multiple of learning rate in projection network')
    parser.add_argument('--fc_weight', type=float, default=10.0, help='multiple of learning rate in the last fc layer of classifier')
    parser.add_argument('--disc_weight', type=float, default=10.0, help='multiple of learning rate in doamin discriminator')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')

    args = parser.parse_args()


    main(args)