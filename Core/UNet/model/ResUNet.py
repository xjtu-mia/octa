import torch
import torch.nn as nn
import torch.utils.data
from Model.modules import (
    ResidualConv,
    ASPP,
    AttentionBlock,
    Upsample_,
    Squeeze_Excite_Block,
)


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=True, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class ResBasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, residual=None, group=1,
                 dilation=1, norm_layer=None):
        super(ResBasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.conv1 = conv3x3(in_planes, out_planes, stride)
        self.bn1 = norm_layer(out_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_planes, out_planes)
        self.bn2 = norm_layer(out_planes)
        self.residual = residual
        self.stride = stride

    def forward(self, x):
        identify = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.residual is not None:
            identify = self.residual(x)

        out += identify
        out = self.relu(out)

        return out


class ResUNet1(nn.Module):
    def __init__(self, channel, num_class=3):
        super(ResUNet1, self).__init__()
        filters = [32, 64, 128, 256, 512]
        self.input_layer = nn.Sequential(
            conv3x3(channel, filters[0]),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(inplace=True),
            conv3x3(filters[0], filters[0]),
            nn.BatchNorm2d(filters[0])
        )

        self.layer2 = self._res_layer(filters[0], filters[1], stride=2)
        self.layer3 = self._res_layer(filters[1], filters[2], stride=2)
        self.layer4 = self._res_layer(filters[2], filters[3], stride=2)
        self.layer5 = self._res_layer(filters[3], filters[4], stride=2)

        self.up5 = self._up_conv(filters[4], filters[3])
        self.layer6 = self._res_layer(filters[4], filters[3])
        self.up6 = self._up_conv(filters[3], filters[2])
        self.layer7 = self._res_layer(filters[3], filters[2])
        self.up7 = self._up_conv(filters[2], filters[1])
        self.layer8 = self._res_layer(filters[2], filters[1])
        self.up8 = self._up_conv(filters[1], filters[0])
        self.layer9 = self._res_layer(filters[1], filters[0])

        self.out_layer = conv1x1(filters[0], num_class)

    def forward(self, x):
        # encoder
        e1 = self.input_layer(x)
        e2 = self.layer2(e1)
        e3 = self.layer3(e2)
        e4 = self.layer4(e3)
        e5 = self.layer5(e4)

        # decoder
        d6 = self.layer6(torch.cat((e4, self.up5(e5)), dim=1))
        d7 = self.layer7(torch.cat((e3, self.up6(d6)), dim=1))
        d8 = self.layer8(torch.cat((e2, self.up7(d7)), dim=1))
        d9 = self.layer9(torch.cat((e1, self.up8(d8)), dim=1))

        return self.out_layer(d9)

    def _res_layer(self, in_planes, out_planes, stride=1, block_num=1):
        residual = nn.Sequential(
            conv1x1(in_planes, out_planes, stride),
            nn.BatchNorm2d(out_planes)
        )
        layers = []
        layers.append(ResBasicBlock(in_planes, out_planes, stride, residual=residual))

        # 下面这一块在block_num=1时是无效的
        for _ in range(1, block_num):
            layers.append(ResBasicBlock(out_planes, out_planes))
        return nn.Sequential(*layers)

    def _up_conv(self, in_planes, out_planes):
        return nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True)
            )


class ResUNet1_ASPP(nn.Module):
    def __init__(self, channel, num_class=3):
        super(ResUNet1_ASPP, self).__init__()
        filters = [32, 64, 128, 256, 512]
        self.input_layer = nn.Sequential(
            conv3x3(channel, filters[0]),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(inplace=True),
            conv3x3(filters[0], filters[0]),
            nn.BatchNorm2d(filters[0])
        )

        self.layer2 = self._res_layer(filters[0], filters[1], stride=2)
        self.layer3 = self._res_layer(filters[1], filters[2], stride=2)
        self.layer4 = self._res_layer(filters[2], filters[3], stride=2)
        # self.layer5 = self._res_layer(filters[3], filters[4], stride=2)
        self.layer5 = ASPP(filters[3], filters[4])

        self.up5 = self._up_conv(filters[4], filters[3])
        self.layer6 = self._res_layer(filters[4], filters[3])
        self.up6 = self._up_conv(filters[3], filters[2])
        self.layer7 = self._res_layer(filters[3], filters[2])
        self.up7 = self._up_conv(filters[2], filters[1])
        self.layer8 = self._res_layer(filters[2], filters[1])
        self.up8 = self._up_conv(filters[1], filters[0])
        self.layer9 = self._res_layer(filters[1], filters[0])

        self.out_layer = conv1x1(filters[0], num_class)

    def forward(self, x):
        # encoder
        e1 = self.input_layer(x)
        e2 = self.layer2(e1)
        e3 = self.layer3(e2)
        e4 = self.layer4(e3)
        e5 = self.layer5(e4)

        # decoder
        d6 = self.layer6(torch.cat((e4, self.up5(e5)), dim=1))
        d7 = self.layer7(torch.cat((e3, self.up6(d6)), dim=1))
        d8 = self.layer8(torch.cat((e2, self.up7(d7)), dim=1))
        d9 = self.layer9(torch.cat((e1, self.up8(d8)), dim=1))

        return self.out_layer(d9)

    def _res_layer(self, in_planes, out_planes, stride=1, block_num=1):
        residual = nn.Sequential(
            conv1x1(in_planes, out_planes, stride),
            nn.BatchNorm2d(out_planes)
        )
        layers = []
        layers.append(ResBasicBlock(in_planes, out_planes, stride, residual=residual))

        # 下面这一块在block_num=1时是无效的
        for _ in range(1, block_num):
            layers.append(ResBasicBlock(out_planes, out_planes))
        return nn.Sequential(*layers)

    def _up_conv(self, in_planes, out_planes):
        return nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True)
            )

class ResUNet2(nn.Module):
    def __init__(self, channel, num_class=3):
        super(ResUNet2, self).__init__()
        filters = [32, 64, 128, 256, 512]
        self.input_layer = nn.Sequential(
            conv3x3(channel, filters[0]),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(inplace=True),
            conv3x3(filters[0], filters[0]),
            nn.BatchNorm2d(filters[0])
        )

        self.layer2 = self._res_layer(filters[0], filters[1], stride=2)
        self.layer3 = self._res_layer(filters[1], filters[2], stride=2)
        self.layer4 = self._res_layer(filters[2], filters[3], stride=2)
        self.layer5 = self._res_layer(filters[3], filters[4], stride=2)

        self.up5 = self._up_conv(filters[4], filters[3])
        self.layer6 = self._res_layer(filters[4], filters[3])
        self.up6 = self._up_conv(filters[3], filters[2])
        self.layer7 = self._res_layer(filters[3], filters[2])
        self.up7 = self._up_conv(filters[2], filters[1])
        self.layer8 = self._res_layer(filters[2], filters[1])
        self.up8 = self._up_conv(filters[1], filters[0])
        self.layer9 = self._res_layer(filters[1], filters[0])

        self.out_layer = conv1x1(filters[0], num_class)

    def forward(self, x):
        # encoder
        e1 = self.input_layer(x)
        e2 = self.layer2(e1)
        e3 = self.layer3(e2)
        e4 = self.layer4(e3)
        e5 = self.layer5(e4)

        # decoder
        d6 = self.layer6(torch.cat((e4, self.up5(e5)), dim=1))
        d7 = self.layer7(torch.cat((e3, self.up6(d6)), dim=1))
        d8 = self.layer8(torch.cat((e2, self.up7(d7)), dim=1))
        d9 = self.layer9(torch.cat((e1, self.up8(d8)), dim=1))

        return self.out_layer(d9)

    def _res_layer(self, in_planes, out_planes, stride=1, block_num=2):
        residual = nn.Sequential(
            conv1x1(in_planes, out_planes, stride),
            nn.BatchNorm2d(out_planes)
        )
        layers = []
        layers.append(ResBasicBlock(in_planes, out_planes, stride, residual=residual))

        # 下面这一块在block_num=1时是无效的
        for _ in range(1, block_num):
            layers.append(ResBasicBlock(out_planes, out_planes))
        return nn.Sequential(*layers)

    def _up_conv(self, in_planes, out_planes):
        return nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True)
            )
