'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
# %%
import torch
import torch.nn as nn
# from torchstat import stat
# from torchsummary import summary
import torch.nn.functional as F


class BasicBlock(nn.Module):  # 18和34层网络
    expansion = 1  # 残差结构的主分支上的卷积层的核的个数（一样==1）

    def __init__(self, in_planes, planes, stride=1):  # stride==1实现shortcut, ==2虚线shortcut
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class BasicBlockV2(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, is_normal=True):
        super(BasicBlockV2, self).__init__()
        self.is_normal = is_normal
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.shortcut = nn.Sequential()

        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.BatchNorm2d(in_planes),
                nn.ReLU(),
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
            )

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        if self.is_normal:
            out += self.shortcut(x)
        else:
            out += 1.5 * self.shortcut(x)
        return out


class Bottleneck(nn.Module):  # 52层及以上
    expansion = 4  # 卷积核个数变化

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class BottleneckV2(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(BottleneckV2, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.BatchNorm2d(in_planes),
                nn.ReLU(),
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
            )

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(self.bn3(out))
        out += self.shortcut(x)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):  # num_blocks列表，对应不同block的个数
        super(ResNet, self).__init__()
        self.in_planes = 64  # maxpooling之后特征矩阵的输入

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)  # conv2_x
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)  # conv3_x
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet12(num_classes=10):
    return ResNet(BasicBlock, [1, 1, 2, 1], num_classes=num_classes)


def ResNet18(num_classes=10):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)


def ResNet34(num_classes=10):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes)


def ResNet50(num_classes=10):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes)


def ResNet101(num_classes=10):
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes)


def ResNet152(num_classes=10):
    return ResNet(Bottleneck, [3, 8, 36, 3], num_classes=num_classes)


def ResNet176(num_classes=10):
    return ResNet(Bottleneck, [3, 12, 40, 3], num_classes=num_classes)


def ResNet200(num_classes=10):
    return ResNet(Bottleneck, [3, 12, 48, 3], num_classes=num_classes)

# def R100esNet12():
#     return ResNet(BasicBlock, [1, 1, 2, 1], 100)

# def R100esNet18():
#     return ResNet(BasicBlock, [2, 2, 2, 2], 100)

# def R100esNet34():
#     return ResNet(BasicBlock, [3, 4, 6, 3], 100)

# def R100esNet50():
#     return ResNet(Bottleneck, [3, 4, 6, 3], 100)

# def R100esNet101():
#     return ResNet(Bottleneck, [3, 4, 23, 3], 100)
#
# def R100esNet152():
#     return ResNet(Bottleneck, [3, 8, 36, 3], 100)
#
# def R100esNet176():
#     return ResNet(Bottleneck, [3, 12, 40, 3], 100)
#
# def R100esNet200():
#     return ResNet(Bottleneck, [3, 12, 48, 3], 100)


# def profile():
#     device = torch.device( "cuda:0" if torch.cuda.is_available() else "cpu" )
#     net = ResNet50()
#     net.to( device )
#     model_name = 'ResNet50'

#     # 模型性能瓶颈分析
#     with torch.autograd.profiler.profile(enabled=True, use_cuda=True, record_shapes=False, profile_memory=False) as prof:
#         y = net(torch.randn(1,3,32,32).to( device ))
#     print(prof.table())
#     prof.export_chrome_trace('./profile/{}.json'.format(model_name))

#     print(y.cpu().detach().numpy().shape)

# def test():
#     net = ResNet18()
#     model_name = 'ResNet18'

#     y = net(torch.randn(1,3,32,32))

#     print( model_name )
#     print(y.shape)

# def show_param():
#     model_name = "ResNet18"
#     net = ResNet200()
#     # net = net.cuda()
#     # stat(net, (3,32,32))
#     summary(net, (3,32,32), device='cpu')

#     print( model_name )
# show_param()
# test()
# profile()

# %%
