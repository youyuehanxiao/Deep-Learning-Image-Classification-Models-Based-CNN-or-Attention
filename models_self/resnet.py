import torch.nn as nn
import torch


class BasicBlock(nn.Module):
    '''
    1. 基本块，非瓶颈结构，没有升维和降维操作
    2. 包含两层卷积层，每一层的卷积核尺寸都是3×3，第二层步长固定为1
    3. 每一层之后都进行批归一化
    '''

    expansion = 1 #通道扩展倍数（瓶颈结构大于1）

    def __init__(self, in_channel, out_channel, stride=1, downsample=None, **kwargs):
        super(BasicBlock, self).__init__()
        #第一层
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, stride=stride, padding=1, bias=False) #卷积对象实例
        self.bn1 = nn.BatchNorm2d(out_channel) #批归一化实例
        self.relu = nn.ReLU() #激活实例
        #第二层
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)

        self.downsample = downsample #进行残差连接时，对原始输入特征所做的操作（为了保证与输出一致），为None表示不做任何操作

    def forward(self, x):
        identity = x
        #是否需要对输入进行处理（最后会与卷积输出相加）
        if self.downsample is not None:
            identity = self.downsample(x)

        #第一层计算
        # print('first_input_size:', x.shape)
        # print('first_weight_size:', self.conv1.weight.shape)
        out = self.conv1(x) #卷积操作
        #print('first_out_size:', out.shape)
        out = self.bn1(out) #批归一化操作
        out = self.relu(out) #激活操作

        #第二层
        # print('second_input_size:', out.shape)
        # print('second_weight_size:', self.conv2.weight.shape)
        out = self.conv2(out)
        #print('second_out_size:', out.shape)
        out = self.bn2(out)

        #残差连接
        out += identity
        #激活
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    '''
    1. 瓶颈结构块，通过点卷积进行将维和升维
    2. 一共三层，第一层点卷积减少通道数（减少计算量），第二层进行标准卷积，第三层点卷积增加通道数（提取更多的特征）
    3. 每一层之后都有批归一化
    '''
    """
    注意:原论文中,在虚线残差结构的主分支上,第一个1x1卷积层的步距是2,第二个3x3卷积层步距是1。
    但在pytorch官方实现过程中是第一个1x1卷积层的步距是1,第二个3x3卷积层步距是2,
    这么做的好处是能够在top1上提升大概0.5%的准确率。
    可参考Resnet v1.5 https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch
    """
    expansion = 4

    def __init__(self, in_channel, out_channel, stride=1, downsample=None,
                 groups=1, width_per_group=64):
        super(Bottleneck, self).__init__()

        width = int(out_channel * (width_per_group / 64.)) * groups #分组处理

        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=width,  kernel_size=1, stride=1, bias=False)  # squeeze channels（减少通道数）
        self.bn1 = nn.BatchNorm2d(width)
        # -----------------------------------------
        self.conv2 = nn.Conv2d(in_channels=width, out_channels=width, groups=groups, kernel_size=3, stride=stride, bias=False, padding=1) #卷积
        self.bn2 = nn.BatchNorm2d(width)
        # -----------------------------------------
        self.conv3 = nn.Conv2d(in_channels=width, out_channels=out_channel*self.expansion, kernel_size=1, stride=1, bias=False)  # unsqueeze channels（增加通道数）
        self.bn3 = nn.BatchNorm2d(out_channel*self.expansion)
        self.relu = nn.ReLU(inplace=True) #激活

        self.downsample = downsample

    def forward(self, x):
        identity = x
        #对输入进行处理，保证与输出一致
        if self.downsample is not None:
            identity = self.downsample(x)

        #第一层，降维
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        #第二层，卷积计算
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        #第三层，升维
        out = self.conv3(out)
        out = self.bn3(out)

        #残差连接
        out += identity

        #激活操作
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    '''
    1. ResNet类，用于构建不同层的ResNet模型
    2. blocks_num为一个序列，包含4个整型元素，每一个元素表示对应基本块的个数。通过设置blocks_num可以构建不同层数的ResNet网络
    '''

    def __init__(self,
                 block,
                 blocks_num,
                 num_classes=1000,
                 include_top=True,
                 groups=1,
                 width_per_group=64):
        '''
        :param block: 基础块（BasicBlock或Bottleneck）
        :param blocks_num: 一个序列，设置每一个基础块使用的个数，以此创建不同层数的网络
        '''
        super(ResNet, self).__init__()
        self.include_top = include_top
        self.in_channel = 64

        self.groups = groups
        self.width_per_group = width_per_group

        #首部处理，减小输入特征尺寸，通道扩展到64
        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2,  padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True) #inplace表示是否将原来值直接替换为激活后的值（不用中间变量进行存储）
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        #创建中间卷积层
        self.layer1 = self._make_layer(block, 64, blocks_num[0])
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)

        #全连接，进行评估
        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  #output size = (1, 1) #自适应平均池化
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        #权重初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _make_layer(self, block, channel, block_num, stride=1):
        downsample = None #输入处理函数

        #输入通道与输出通道不相等，对输入通道进行增加或减少处理（保证正确地进行残差连接）
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion))

        layers = []
        #当前layer的第一个基础块
        layers.append(block(self.in_channel,
                            channel,
                            downsample=downsample,
                            stride=stride,
                            groups=self.groups,
                            width_per_group=self.width_per_group))

        #第一个基础块结束后，其输出为下一个基础块的输入（后面几个块中的第一层输入的通道数均相同）
        self.in_channel = channel * block.expansion

        for _ in range(1, block_num):
            layers.append(block(self.in_channel,
                                channel,
                                groups=self.groups,
                                width_per_group=self.width_per_group))

        #将当前layer中的块按顺序连接在一起（上一个块的输出为下一个块的输入）
        return nn.Sequential(*layers)

    def forward(self, x):
        #首部处理
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        #卷积层计算
        x = self.layer1(x)
        #print('*'*50)
        x = self.layer2(x)
        #print('*' * 50)
        x = self.layer3(x)
        #print('*' * 50)
        x = self.layer4(x)
        #print('*' * 50)

        #评估打分
        if self.include_top:
            x = self.avgpool(x)
            #print(x.shape)
            x = torch.flatten(x, 1) #前1维不变，后面维度展平
            x = self.fc(x)
            #print(x.shape)

        return x

# # resnet34  pre-train parameters https://download.pytorch.org/models/resnet34-333f7ec4.pth
# def resnet_samll(num_classes=1000, include_top=True):
   
#     return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)

# # resnet50  pre-train parameters https://download.pytorch.org/models/resnet50-19c8e357.pth
# def resnet(num_classes=1000, include_top=True): 
#     return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)

# # resnet101 pre-train parameters https://download.pytorch.org/models/resnet101-5d3b4d8f.pth
# def resnet_big(num_classes=1000, include_top=True):
#     return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, include_top=include_top)

# # resneXt pre-train parameters https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth
# def resnext(num_classes=1000, include_top=True): 
#     groups = 32
#     width_per_group = 4
#     return ResNet(Bottleneck, [3, 4, 6, 3],
#                   num_classes=num_classes,
#                   include_top=include_top,
#                   groups=groups,
#                   width_per_group=width_per_group)

# # resneXt_big pre-train parameters https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth
# def resnext_big(num_classes=1000, include_top=True): 
#     groups = 32
#     width_per_group = 8
#     return ResNet(Bottleneck, [3, 4, 23, 3],
#                   num_classes=num_classes,
#                   include_top=include_top,
#                   groups=groups,
#                   width_per_group=width_per_group)

def resnet34(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnet34-333f7ec4.pth
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)


def resnet50(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnet50-19c8e357.pth
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)


def resnet101(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnet101-5d3b4d8f.pth
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, include_top=include_top)


def resnext50_32x4d(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth
    groups = 32
    width_per_group = 4
    return ResNet(Bottleneck, [3, 4, 6, 3],
                  num_classes=num_classes,
                  include_top=include_top,
                  groups=groups,
                  width_per_group=width_per_group)


def resnext101_32x8d(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth
    groups = 32
    width_per_group = 8
    return ResNet(Bottleneck, [3, 4, 23, 3],
                  num_classes=num_classes,
                  include_top=include_top,
                  groups=groups,
                  width_per_group=width_per_group)

if __name__ == '__main__':
    imgs_1 = torch.rand((5, 3, 224, 224))
    #print(imgs_1)
    #imgs_2 = torch.rand((5, 1, 2, 2))
    #print(imgs_2)
    #imgs_s = torch.cat([imgs_1, imgs_2], dim=1)
    # print(imgs_s.shape)
    # print(imgs_s)
    bc = resnet34(8)
    bc(imgs_1)
