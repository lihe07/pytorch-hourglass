"""
https://github.com/lihe07/pytorch-hourglass/

StackedHourglass 模型实现
# With annotations
By @lihe07 2021.11.22
"""

import torch
from torch import nn

from icecream import ic


# Concepts
# 中间监督:
# 在网络内部评估网络
# MSE:
# 均方误差

# 小模块
def make_relu_conv(in_channels, out_channels, kernel_size=1, stride=1, padding=0):
    """
    制造一个激活函数为Relu的卷积层
    :param in_channels: 输入通道数
    :param out_channels: 输出通道数
    :param kernel_size: 核心大小
    :param stride: 步长
    :param padding: 周围填充
    :return: 一个Sequential
    """

    # print('debug', in_channels, out_channels, kernel_size, stride, padding)
    return nn.Sequential(
        nn.BatchNorm2d(in_channels),
        nn.Conv2d(in_channels, out_channels, (kernel_size, kernel_size), (stride, stride), padding),
        nn.ReLU(inplace=True)
    )


def make_conv_group(in_channels, out_channels):
    """
    制造一窝卷积层
    Data [x, y, in_channels]
    -> ReluConv(size=1, stride=1, padding=0) -> Data [x, y, out_channels//2]
    -> ReluConv(size=3, stride=1, padding=1) -> Data [x, y, out_channels//2]
    -> ReluConv(size=1, stride=1, padding=0) -> Data [x, y, out_channels]
    :param in_channels: 输入的通道数量
    :param out_channels: 输出通道数量
    :return: 一个Sequential 包含三个ReluConv
    """
    middle_channels = out_channels // 2
    if middle_channels == 0:
        middle_channels = 1
    return nn.Sequential(
        make_relu_conv(in_channels, middle_channels, 1, 1, 0),
        make_relu_conv(middle_channels, middle_channels, 3, 1, 1),
        make_relu_conv(middle_channels, out_channels, 1, 1, 0)
        # 内核大小为1的卷积层可用来调整channels
    )


def make_channel_controller(in_channels, out_channels):
    """
    制造一个通道调节器
    Data [x, y, in_channels]
    -> Conv(size=1, stride=1, padding=0) -> Data [x, y, out_channels]
    :param in_channels: 输入的通道数量
    :param out_channels: 输出的通道数量
    :return: 一个Sequential 可能包含一个Conv
    """
    if in_channels == out_channels:
        return nn.Sequential()
    else:
        return nn.Conv2d(in_channels, out_channels, (1, 1))


class ResBlock(nn.Module):
    """
    残差模块
    """

    def __init__(self, in_channels, out_channels):
        super(ResBlock, self).__init__()
        self.conv = make_conv_group(in_channels, out_channels)
        self.skip = make_channel_controller(in_channels, out_channels)
        # debug
        # self.inc = in_channels

    def forward(self, x):
        # print("输入x的shape为", x.shape)
        # print("输入的channel应为", self.inc)

        return self.conv(x) + self.skip(x)


def make_res_block(in_channels, out_channels):
    """
    制造一个残差模块
    Data [x, y, in_channels] -> ChannelController() -
    |                                               |
    -> ConvBlock() -> Data [x, y, out_channels] -> (+) -> Data [x, y, out_channels]
    :param in_channels: 输入的通道数量
    :param out_channels: 输出的通道数量
    :return: 一个nn.Module 内含一个残差模块
    """
    return ResBlock(in_channels, out_channels)


class Enlarger(nn.Module):
    def __init__(self, kernel):
        super(Enlarger, self).__init__()
        self.kernel = kernel
        # 缩放比例

    def forward(self, x):
        return x.kron(torch.ones(self.kernel))


def make_enlarger(zoom_factor):
    """
    制造一个图像放大器
    将图片水平垂直放大两倍
    -------      -------------
    |  *  |  => |    *  *    |
    -------     |    *  *    |
                --------------

    Wout = Win * zoom_factor
    Hout = Hin * zoom_factor
    :param zoom_factor: 放大倍数
    :return: 一个nn.Module 用于放大图片
    """
    # 有待实验
    # return Enlarger((zoom_factor, zoom_factor))
    return nn.Upsample(scale_factor=zoom_factor)


def make_res_group(channels, num):
    """
    制造一窝残差网络
    :param channels: 输入通道数量==输出通道数量
    :param num: 包含多少个残差
    :return: 一个nn.Sequential
    """
    layers = []
    for _ in range(num):
        layers.append(make_res_block(channels, channels))
    return nn.Sequential(*layers)


# 主网络部分

class Hourglass(nn.Module):
    """
    单个沙漏模型
    ===  判断 nesting_times 大小  ===
    # [ >1 ]
    Data -> MaxPool -> [ res_block(channels, channels) * n ]
      |  -> Hourglass( nesting_times-=1 )
 skip |  -> [ res_block(channels, channels) * n ]
      |  -> Enlarger(k) |
     ------------------(+)
                        |
                       Data
    # [ =1 ]
    Data -> MaxPool -> [ res_block(channels, channels) * n ]
     |   -> [ res_block(channels, channels) * n ]
skip |   -> [ res_block(channels, channels) * n ]
     |   -> Enlarger(k)  |
    --------------------(+)
                         |
                        Data
    # skipper:
    Data -> [ res_block(channels, channels) * n ] -> Data
    """

    def __init__(self, channels, group_res_num, nesting_times, pool_kernel=2, pool_stride=2, enlarger_kernel=2):
        """
        构造一个大沙漏模型
        :param channels: 输入==输出通道数量
        :param group_res_num: 一窝残差包含的残差Conv数量
        :param nesting_times: 递归套娃多少个沙漏
        :param pool_kernel: Max池化层的核心大小
        :param pool_stride: Max池化层的步长
        :param enlarger_kernel: 图片放大器放大倍数
        """
        super(Hourglass, self).__init__()

        # 制造skipper
        self.skipper = make_res_group(channels, group_res_num)

        # 制造主线
        self.main_route = nn.Sequential(
            nn.MaxPool2d(kernel_size=pool_kernel, stride=pool_stride),
            make_res_group(channels, group_res_num),

            # 分支1: 如果还有递归次数
            Hourglass(channels,
                      group_res_num,
                      nesting_times - 1,
                      pool_kernel,
                      pool_stride,
                      enlarger_kernel
                      )
            if nesting_times > 1
            else make_res_group(channels, group_res_num),  # 分支2: 没有递归次数

            make_res_group(channels, group_res_num),
            make_enlarger(enlarger_kernel)
        )

    def forward(self, x):
        # 每次Nesting会导致size的W和H减半
        assert x.shape[2] % 2 == 0 and x.shape[3] % 2 == 0, RuntimeError(
            f"每次Nesting都会导致高度和宽度减半, 目前输入的高度和宽度 {x.shape[3]} {x.shape[2]} 无法被2整除")
        # x = self.main_route(x)
        # s = self.skipper(x)
        # ic(x.shape, s.shape)
        # if x.shape != s.shape:
        #     ic(x.shape, s.shape)
        #     x = nn.Upsample((
        #         s.shape[2],
        #         s.shape[3]
        #     ))(x) + s
        # else:
        #     x = x + s
        return self.main_route(x) + self.skipper(x)


class ConnectedHourglass(nn.Module):
    """
    叠在一起的沙漏模型
    Data [3] -> ReluConv(7, 2, 3) -> Data [64]
        -> res_block(64, 128)
        -> MaxPool(2, 2)
        -> res_block(128, 128)
        -> res_block(128, channels)
        -> Data [channels]
    # 开始循环
    # 单个循环结
    Data [channels] -> Hourglass
    |    -> [ res_block(channels, channels) * n ]
    |    -> ReluConv() -> ChC -> Out [ i ]
    |            |          |
    |ChC(channels) ->(+)<- ChC(channels)
    |                 |
    |               Data [ channels ]
    |----------------(+)
                    Data [ channels ]
    """

    def __init__(self, channels, hourglass, stack_num, group_res_num, heatmap_dimensions):
        """
        :param channels: 输入的通道数量
        :param hourglass: 创建好的沙漏网络
        :param heatmap_dimensions: 输出几维度的Heatmap
        """
        super(ConnectedHourglass, self).__init__()
        self.top = nn.Sequential(
            make_relu_conv(3, 64, 7, 2, 3),
            make_res_block(64, 128),
            nn.MaxPool2d(2, 2),
            make_res_block(128, 128),
            make_res_block(128, channels)
        )

        # 循环节的前部分
        # Contains hourglass + res_group + relu_conv
        self.loop_head = nn.ModuleList([
            nn.Sequential(
                hourglass,
                make_res_group(channels, group_res_num),
                make_relu_conv(channels, channels)
            )
            for _ in range(stack_num)
        ])
        # 接受原始输出生成Heatmap的ChC
        self.origin_to_heatmap_chc = nn.ModuleList([
            make_channel_controller(channels, heatmap_dimensions)
            for _ in range(stack_num)
        ])
        # 接受原始输出生成Data一部分的ChC
        self.origin_to_data_chc = nn.ModuleList([
            make_channel_controller(channels, channels)
            for _ in range(stack_num)
        ])
        # 接受Heatmap,生成Data一部分的ChC
        self.heatmap_to_data_chc = nn.ModuleList([
            make_channel_controller(heatmap_dimensions, channels)
            for _ in range(stack_num)
        ])

    def forward(self, x):
        """
        沙漏堆叠的前向传播稍微复杂
        :param x:
        :return: out
        """
        out = torch.Tensor()
        # print(f"[DEBUG] 堆叠网络收到了 {x.shape} 大小的数据")
        # assert len(x.shape) == 4, ValueError("输入大小需为 [N, C, W, H]")
        # assert x.shape[1] == 3, ValueError("输入通道数需为 3 [N, C, W, H]")
        x = self.top(x)
        ic(x.shape)
        for i, head in enumerate(self.loop_head):
            # 遍历全部的循环节
            # 每一个循环节会生成一个Heatmap
            ic("HG输入", x.shape)
            origin = head(x)
            ic("HG输出", origin.shape)

            heatmap = self.origin_to_heatmap_chc[i](origin)
            # ic(self.origin_to_data_chc[i](origin).shape, self.heatmap_to_data_chc[i](heatmap).shape)
            data = self.origin_to_data_chc[i](origin) + self.heatmap_to_data_chc[i](heatmap)

            out = torch.stack((out, heatmap))
            x += data

        return out


def make_connected_hourglass(channels, group_res_num, nesting_times, heatmap_dimensions, stack_num, pool_kernel=2,
                             pool_stride=2, enlarger_kernel=2):
    """
    制造一个叠起来的沙漏
    :param channels: 通道数量
    :param group_res_num: 一个残差小组包含的残差网络数量
    :param nesting_times: 漏斗套娃数量
    :param heatmap_dimensions: 输出的Heatmap维度
    :param stack_num: 堆叠几个沙漏
    :param pool_kernel: 池化层核心
    :param pool_stride: 池化层步长
    :param enlarger_kernel: 放大器倍数
    :return: 一个沙漏网络
    """
    hourglass = Hourglass(channels, group_res_num, nesting_times, pool_kernel, pool_stride, enlarger_kernel)
    return ConnectedHourglass(channels, hourglass, stack_num, group_res_num, heatmap_dimensions)


class HeatmapLoss(nn.Module):
    """
    Heatmap损失函数

    """

    def __init__(self):
        super(HeatmapLoss, self).__init__()


def test_train():
    from torch.optim import SGD

    print("训练测试开始")
    model = make_connected_hourglass(16, 4, 2, 1, 2)
    x = torch.rand((1, 3, 256, 256))
    y = torch.rand([1, 2, 1, 64, 64])
    out = model(x)
    optim = SGD(model.parameters(), lr=0.001, momentum=0.8)
    lossfunc = nn.MSELoss()
    ic(out.shape)
    loss = lossfunc(out, y)
    optim.zero_grad()
    loss.backward()
    optim.step()


if __name__ == '__main__':
    # TODO: 逐步debug model
    with torch.autograd.detect_anomaly():
        test_train()
