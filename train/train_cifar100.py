"""Train CIFAR10 with PyTorch."""
import datetime as dt
import os
import time
from datetime import datetime

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
# from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import _LRScheduler

from utils import myargs, selectarch, selectopt

import sys

sys.path.append(".")  

import optimizers as opt
# Training
trainName = 'PyTorch CIFAR10 Training'
args = myargs(trainName)
args.save_path = './runs/cifar100/' + args.arch + '/Ori' + \
                 '_BS' + str(args.batch_size) + 'LR' + \
                 str(args.lr) + 'epoch' + \
                 str(args.epoch) + 'warmup' + str(args.warm) + \
                 args.notes + \
                 "{0:%Y-%m-%dT%H-%M/}".format(datetime.now())


def train(epoch: int):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    end = time.time()

    net.train()

    for step, (inputs, targets) in enumerate(trainloader):
        # if epoch <= args.warm:
        # warmup_scheduler.step()

        # [batch_idx==step][(inputs, targets)==data]
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad() #reset the gradients of model parameters. Gradients by default add up; to prevent double-counting, we explicitly zero them at each iteration.
        outputs = net(inputs)
        loss = loss_function(outputs, targets)
        acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(acc1[0], inputs.size(0))
        top5.update(acc5[0], inputs.size(0))

        loss.backward() 
        if args.sub_epoch != 0:
            optimizer.step(epoch=epoch,kappa=args.kappa, sub_epoch=args.sub_epoch)
        else:
            optimizer.step() 

        batch_time.update(time.time() - end)
        end = time.time()

    print('Epoch: {}'.format(epoch))
    print('Train set,', end=' ')
    print('Average loss: {:.4f}, Accuracy: {:.4f}'.format(losses.avg, top1.avg), end='<==>')

    return top1.avg, losses.avg, batch_time.sum


def test(epoch: int):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    end = time.time()

    net.eval()

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)  # inputs==images, targets==labels
            outputs = net(inputs)
            loss = loss_function(outputs, targets)
            acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(acc1[0], inputs.size(0))
            top5.update(acc5[0], inputs.size(0))
            batch_time.update(time.time() - end)
            end = time.time()

    print('Test set,', end=' ')
    print('Average loss: {:.4f}, Accuracy: {:.4f}'.format(losses.avg, top1.avg))

    

    return top1.avg, losses.avg, batch_time.sum

class AverageMeter(object):
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


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            # correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """

    def __init__(self, optimizer, total_iters, last_epoch=-1):
        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]

if __name__ == '__main__':

    print("****{}****".format(trainName))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("using {} device.".format(device))

    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    checkpoint_interval = 100  
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),

    ])

    trainset = torchvision.datasets.CIFAR100(
        root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, args.batch_size, shuffle=True, num_workers=args.workers)
    testset = torchvision.datasets.CIFAR100(
        root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, args.batch_size, shuffle=False, num_workers=args.workers)
    train_num = len(trainset)
    val_num = len(testset)
    print("using {} images for training, {} images for validation.".format(train_num, val_num))

    print('==> Building model {}'.format(args.arch))
    print('==> Train info:')
    print('==> Start DateTime: {}'.format(dt.datetime.now()))


    net = selectarch(args.arch, num_classes=100)
    net = net.to(device)

    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    loss_function = nn.CrossEntropyLoss()
    optimizer = selectopt(net, argsopt=args.opt, lr=args.lr, m=args.m, m2=args.m2, sub_epoch=args.sub_epoch,
                          weight_decay=args.weight_decay)

    train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 75], gamma=0.1)


    train_time = 0.0
    test_time = 0.0
    train_top1_acc = 0.0
    train_min_loss = 100
    test_top1_acc = 0.0
    test_min_loss = 100


    print('==> Training epoch:{}'.format(args.epoch))
    for epoch in range(1, args.epoch+1):

        train_acc_epoch, train_loss_epoch, train_epoch_time = train(epoch)
        train_top1_acc = max(train_top1_acc, train_acc_epoch)
        train_min_loss = min(train_min_loss, train_loss_epoch)
        train_time += train_epoch_time
        acc, test_loss_epoch, test_epoch_time = test(epoch)
        test_top1_acc = max(test_top1_acc, acc)
        test_min_loss = min(test_min_loss, test_loss_epoch)
        test_time += test_epoch_time
        train_scheduler.step()
       
    end_train = train_time // 60
    end_test = test_time // 60
    print(args.arch)
    print("train time: {}D {}H {}M".format(end_train // 1440, (end_train % 1440) // 60, end_train % 60))
    print("tset time: {}D {}H {}M".format(end_test // 1440, (end_test % 1440) // 60, end_test % 60))
    print("train_acc_top1:{}, train_min_loss:{}, train_time:{}".format(train_top1_acc, train_min_loss, train_time))
    print("test_top1_acc:{}, test_min_loss:{}, test_time:{}".format(test_top1_acc, test_min_loss, test_time))

