'''
一、创建一个输入通道是in_channels、输出通道是out_channels、保留 K×K 尺寸通道数为 1/P 的异构卷积层的主要思路：（HetConv）
1. 总共是要创建out_channels个滤波器，每个滤波器的通道数是in_channels。对于每一个滤波，进行逐通道创建。
2. 对于第i个滤波器，第j个通道有：
    当 j = (i + P * K_count) % in_channels 且 K_count＜in_channels/P，其中K_count为当前滤波器已经创建得到尺寸为K的通道数时，有
    H(ij) = nn.Conv2d(1, 1, kernel_size=K, stride=stride, padding=pd, bias=False) #pd = 1/2 * (K -1) K为奇数
    当j为其它值时，有
    H(ij) = nn.Conv2d(1, 1, kernel_size=1, stride=stride, padding=0, bias=False)
3. 将所有滤波器存放到一个二维列表filters_list中，且元素filters_list[i][j]表示第i个滤波器，第j个通道
4. 在进行卷积运算时，每一个滤波的各个通道分别与输入样本的对应通道进行卷积运算，然后将运算结果进行相加；不同滤波器的卷积结果进行通道拼接；最终得到异构卷积的结果。

二、思路二：（Efficient_HetConv）
异构卷积的结果 = 分P组的K卷积 + 不分组的点卷积 - 分P组的点卷积
'''
import torch.nn as nn
import torch
class HetConv(nn.Module):
    def __init__(self, in_channels:int, out_channels:int, P:int, kernel_size:int, stride:int, bias=False):
        '''
        :param in_channels: 输入通道数
        :param out_channels: 输出通道数（滤波器个数）
        :param P: 每一个滤波器使用 1/P * in_channels个尺寸为Kernel_size的滤波器
        :param kernel_size: 使用的卷积核尺寸（不能于1）
        :param bias:是否使用偏置参数
        '''
        super(HetConv, self).__init__()
        self.filters_list = self.create_filters(in_channels, out_channels, P, kernel_size, stride, bias)

    def create_filters(self, in_channels:int, out_channels:int, P:int, kernel_size:int, stride:int, bias=False) -> nn.ModuleList:
        filters_list = nn.ModuleList() #存储异构卷积滤波器，filters_list[i][j]表示第i个滤波器，第j个通道

        #创建异构卷积滤波器
        for i in range(out_channels):
            filters_list_i = nn.ModuleList()
            #创建第i个滤波器的每个通道（从第i个通道开始，每隔P个通道创建一个尺寸为kernel_size的通道）
            first_K = i % in_channels #第一个尺寸为K的通道序号
            for j in range(in_channels):
                conv = None #当前通道
                if j < first_K:
                    if (in_channels + j - first_K) % P == 0:
                        conv = nn.Conv2d(1, 1, kernel_size=kernel_size, padding=int((kernel_size - 1) / 2),
                                         stride=stride, bias=bias) #创建尺寸为kernel_size的卷积核
                    else:
                        conv = nn.Conv2d(1, 1, kernel_size=1, padding=0, stride=stride, bias=bias) #创建尺寸为1的卷积核
                else:
                    if (j - first_K) % P == 0:
                        conv = nn.Conv2d(1, 1, kernel_size=kernel_size, padding=int((kernel_size - 1) / 2),
                                         stride=stride, bias=bias)  #创建尺寸为kernel_size的卷积核
                    else:
                        conv = nn.Conv2d(1, 1, kernel_size=1, padding=0, stride=stride, bias=bias)  #创建尺寸为1的卷积核

                filters_list_i.append(conv) #将第j个通道加入第i个滤波器

            filters_list.append(filters_list_i) #将第i个滤波器加入列表

        return filters_list

    def forward(self, x):
        out = None #异构卷积结果
        for i, filter_i in enumerate(self.filters_list):
            out_i = None #卷积后的第i个通道
            for j, channel_j in enumerate(filter_i):
                out_j = channel_j(x[:, j:j+1, :, :]) #对第j个通道进行卷积运算
                if j == 0:
                    out_i = out_j
                else:
                    out_i += out_j

            #拼接不同卷积后的不同通道
            if i == 0:
                out = out_i
            else:
                out = torch.cat([out, out_i], dim=1)

        #print('out_size:', out.shape)

        return out

class Efficient_HetConv(nn.Module):
    def __init__(self, in_channels:int, out_channels:int, P:int, kernel_size:int, stride:int, bias=False):
        '''
        :param in_channels: 输入通道数
        :param out_channels: 输出通道数（滤波器个数）
        :param P: 每一个滤波器使用 1/P * in_channels个尺寸为Kernel_size的滤波器
        :param kernel_size: 使用的卷积核尺寸（不能于1）
        :param bias:是否使用偏置参数
        '''
        super(Efficient_HetConv, self).__init__()

        #创建异构卷积滤波器
        if in_channels % P != 0:
            raise ValueError('in_channels must be divisible by p')
        if out_channels % P != 0:
            raise ValueError('out_channels must be divisible by p')

        # 分P组K卷积
        self.conv3x3_P = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                 padding=int((kernel_size - 1) / 2), groups=P, bias=bias)

        # 分P组点卷积
        self.conv1x1_P = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride,
                                 padding=0, groups=P, bias=bias)

        # 不分组点卷积
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride,
                                 padding=0, groups=1, bias=bias)

    def forward(self, x):
        out = self.conv3x3_P(x) + self.conv1x1(x) - self.conv1x1_P(x)
        #print(out.shape)
        return out

if __name__ == '__main__':
    imgs_1 = torch.rand((5, 64, 224, 224))
    # ht = HetConv(64, 128, 128, 5, 2)
    # for i, filter_i in enumerate(ht.filters_list):
    #     print(f'第{i}个滤波器:')
    #     for j, channel_j in enumerate(filter_i):
    #         if channel_j.weight.shape[2] != 1:
    #             print(f'第{j}个通道尺寸：', channel_j.weight.shape)
    #     print('='*50)
    # ht(imgs_1)
    ht2 = Efficient_HetConv(64, 64, 64, 7, 2)
    ht2(imgs_1)


