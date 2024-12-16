import argparse
import torch.optim as optim
import sys

sys.path.append(".")  

import optimizers as opt
import models

def myargs(trainName: str):
    parser = argparse.ArgumentParser(description=trainName)
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    parser.add_argument('--epoch', type=int, default=200, help='training epoch')
    parser.add_argument('--warm', type=int, default=1, help='warm up training phase')
    parser.add_argument('--cifar', type=int, default=10, help='select cifar10 or cifar100 dataset')
    
    parser.add_argument('-d', '--data', default='./data', type=str)
    parser.add_argument('--arch', '-a', default='r18', type=str)
    parser.add_argument('--sub_epoch', '-se', default=0, type=int)
    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')

    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-bs', '--batch_size', default=256, type=int,
                        metavar='N', help='mini-batch size (default: 256)')
    parser.add_argument('--test-batch', default=32, type=int, metavar='N',
                        help='test batchsize (default: 200)')

    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--print-freq', '-p', default=250, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('-c', '--checkpoint', type=str, metavar='PATH',
                        help='path to save checkpoint (default: checkpoint)')

    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='use pre-trained model')
    # Optimization options
    parser.add_argument('--opt_level', default='O2', type=str,
                        help='O2 is fast mixed FP16/32 training, O0 (FP32 training) and O3 (FP16 training), O1 ("conservative mixed precision"), O2 ("fast mixed precision").--opt_level O1 and O2 both use dynamic loss scaling by default unless manually overridden. --opt-level O0 and O3 (the "pure" training modes) do not use loss scaling by default. See more in https://github.com/NVIDIA/apex/tree/f5cd5ae937f168c763985f627bbf850648ea5f3f/examples/imagenet')
    parser.add_argument('--keep-batchnorm-fp32', default=True, action='store_true',
                        help='keeping cudnn bn leads to fast training')
    parser.add_argument('--loss-scale', type=float, default=None)
    parser.add_argument('--dali_cpu', action='store_true',
                        help='Runs CPU based version of DALI pipeline.')
    parser.add_argument('--prof', dest='prof', action='store_true',
                        help='Only run 10 iterations for profiling.')
    parser.add_argument('-t', '--test', action='store_true',
                        help='Launch test mode with preset arguments')
    parser.add_argument('--warmup', '--wp', default=5, type=int,
                        help='number of epochs to warmup')
    parser.add_argument('--weight-decay', '--wd', default=4e-5, type=float,
                        metavar='W', help='weight decay (default: 4e-5 for mobile models)')
    parser.add_argument('--wd-all', dest='wdall', action='store_true',
                        help='weight decay on all parameters')

    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument("--ex", default=0, type=int)
    parser.add_argument("--alpha", default=0.1, type=float)
    parser.add_argument("--beta", default=15.0, type=float)
    parser.add_argument("--notes", default='', type=str)
    parser.add_argument("--m", default=1, type=int)
    parser.add_argument("--opt", default='sgd', type=str)
    parser.add_argument("--m2", default=False, type=bool)
    parser.add_argument("-kappa", default=0, type=float)
    parser.add_argument("-int_num", default=0, type=int)
    parser.add_argument("-noise", default=0, type=float)

    return parser.parse_args()


def selectarch(arch: str, num_classes=10):
    Net = {
        'r12': models.ResNet12(num_classes),
        'r18': models.ResNet18(num_classes),
        'r34': models.ResNet34(num_classes),
        'r50': models.ResNet50(num_classes),
        'r101': models.ResNet101(num_classes),
        'r152': models.ResNet152(num_classes),
        'r176': models.ResNet176(num_classes),
        'r200': models.ResNet200(num_classes),
        'dense121': models.DenseNet121(num_classes),
        
        "PRN": models.PRN.PyramidNet(dataset="cifar10",depth=110, alpha=64, num_classes=10),
        'PRN100': models.PRN.PyramidNet(dataset="cifar100",depth=110, alpha=64, num_classes=100),


    }

    return Net.get(arch)  


def selectopt(net, argsopt: str, lr=1e-3, m=0, m2=False, sub_epoch=0, weight_decay=0, kappa=0):

    if argsopt == 'sgd':
        if m == 1:
            optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
        else:
            optimizer = optim.SGD(net.parameters(), lr=lr)
    elif argsopt == 'adam':
        optimizer = optim.Adam(net.parameters(), lr=lr,betas=(0.9, 0.999),weight_decay=0.5e-4)
    elif argsopt == 'adan':
        optimizer = opt.Adan(net.parameters(), lr=lr, weight_decay=0.5e-4)
    elif argsopt == 'amsgrad':
        optimizer = optim.Adam(net.parameters(), lr=lr,betas=(0.9, 0.999),weight_decay=0.5e-4)
    elif argsopt == 'adamw':
        optimizer = optim.AdamW(net.parameters(), lr=lr,betas=(0.9, 0.999),weight_decay=0.5e-4)
    elif argsopt == 'hadam':
        optimizer = opt.Hadam(net.parameters(), lr=lr,betas=(0.9, 0.999),weight_decay=0.5e-4)
    elif argsopt == 'nadam':
        optimizer = opt.Nadam(net.parameters(), lr=lr,betas=(0.9, 0.999))
    elif argsopt == 'adamax':
        optimizer = optim.Adamax(net.parameters(), lr=lr,betas=(0.9, 0.999),weight_decay=0.5e-4)
    elif argsopt == 'yogi':
        optimizer = opt.Yogi(net.parameters(), lr=lr,betas=(0.9, 0.999),weight_decay=0.5e-4)
    elif argsopt == 'adabound':
        optimizer = opt.AdaBound(net.parameters(), lr=lr,betas=(0.9, 0.999),weight_decay=0.5e-4)
    elif argsopt == 'adabelief':
        optimizer = opt.AdaBelief(net.parameters(), lr=lr,betas=(0.9, 0.999),weight_decay=0.5e-4)
    elif argsopt == 'adafactor':
        optimizer = opt.Adafactor(net.parameters(), lr=lr,betas=(0.9, 0.999),weight_decay=0.5e-4)
    elif argsopt == 'adamp':
        optimizer = opt.AdamP(net.parameters(), lr=lr,betas=(0.9, 0.999),weight_decay=0.5e-4)
    elif argsopt == 'lion':
        optimizer = opt.Lion(net.parameters(), lr=lr,betas=(0.9, 0.999),weight_decay=0.5e-4)
    elif argsopt == 'apollo':
        optimizer = opt.Apollo(net.parameters(), lr=lr,weight_decay=2.5e-4)
    elif argsopt == 'ndadam':
        optimizer = opt.NDAdam(net.parameters(), lr=lr,betas=(0.9, 0.999))
    elif argsopt == 'miadam1':
        optimizer = opt.MIAdam1(net.parameters(), lr=lr,betas=(0.9, 0.999),weight_decay=0.5e-4)     
    elif argsopt == 'miadam2':
        optimizer = opt.MIAdam2(net.parameters(), lr=lr,betas=(0.9, 0.999),weight_decay=0.5e-4) 
    elif argsopt == 'miadam3':
        optimizer = opt.MIAdam3(net.parameters(), lr=lr,betas=(0.9, 0.999),weight_decay=0.5e-4)        
    elif argsopt == 'adai': 
        optimizer = opt.Adai(net.parameters(), lr=lr, betas=(0.1, 0.99), eps=1e-03, weight_decay=5e-4, decoupled=False)
    elif argsopt == 'swats': 
        optimizer = opt.SWATS(net.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=5e-4)
    return optimizer
