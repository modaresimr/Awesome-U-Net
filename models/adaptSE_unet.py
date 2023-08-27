import torch
from torch import nn
from .adapconv import Conv_DCFD


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, with_bn=False):
        super().__init__()
        if with_bn:
            self.step = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
            )
        else:
            self.step = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(),
            )

    def forward(self, x):
        return self.step(x)


class AdaptUNet(nn.Module):
    def __init__(self, in_channels, out_channels, with_bn=False, adaptive_kernel_min_size=3, adaptive_kernel_max_size=3):
        super().__init__()
        init_channels = 32
        self.out_channels = out_channels
        self.adaptive_layer = Conv_DCFD(in_channels, in_channels,
                                        adaptive_kernel_max_size=adaptive_kernel_max_size,
                                        adaptive_kernel_min_size=adaptive_kernel_min_size,
                                        inter_kernel_size=3, padding=1, stride=1, bias=True)  # 0.5% behtar shod
        # self.en_1 = Conv_DCFD(in_channels, init_channels, kernel_size=3, inter_kernel_size=5, padding=1, stride=1, bias=True)  # 0.5% behtar shod
        self.en_1 = DoubleConv(self.adaptive_layer.new_out_channels, init_channels, with_bn)
        self.en_2 = DoubleConv(1 * init_channels, 2 * init_channels, with_bn)
        self.en_3 = DoubleConv(2 * init_channels, 4 * init_channels, with_bn)
        self.en_4 = DoubleConv(4 * init_channels, 8 * init_channels, with_bn)

        self.de_1 = DoubleConv((4 + 8) * init_channels, 4 * init_channels, with_bn)
        self.de_2 = DoubleConv((2 + 4) * init_channels, 2 * init_channels, with_bn)
        self.de_3 = DoubleConv((1 + 2) * init_channels, 1 * init_channels, with_bn)

        self.de_4 = nn.Conv2d(init_channels, out_channels, 1)
        self.adaptive_layerE = Conv_DCFD(out_channels, out_channels,
                                         adaptive_kernel_max_size=adaptive_kernel_max_size,
                                         adaptive_kernel_min_size=adaptive_kernel_min_size,
                                         inter_kernel_size=3, padding=1, stride=1, bias=True)  # 0.5% behtar shod
        self.de_5 = nn.Conv2d(self.adaptive_layerE.new_out_channels, out_channels, 1)

        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self, xx):
        x = self.adaptive_layer(xx)
        # print(xx.shape, x.shape, self.DCFD.new_out_channels, "DDD")
        e1 = self.en_1(x)
        e2 = self.en_2(self.maxpool(e1))
        e3 = self.en_3(self.maxpool(e2))
        e4 = self.en_4(self.maxpool(e3))

        d1 = self.de_1(torch.cat([self.upsample(e4), e3], dim=1))
        d2 = self.de_2(torch.cat([self.upsample(d1), e2], dim=1))
        d3 = self.de_3(torch.cat([self.upsample(d2), e1], dim=1))
        d4 = self.de_4(d3)
        d4 = self.adaptive_layerE(d4)
        d4 = self.de_5(d4)
        return d4

#         if self.out_channels<2:
#             return torch.sigmoid(d4)
#         return torch.softmax(d4, 1)
