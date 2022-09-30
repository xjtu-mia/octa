import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from Model.modules import (
    ResidualConv,
    ASPP,
    ASPP2,
    AttentionBlock,
    Upsample_,
    Squeeze_Excite_Block,
)


class conv_block(nn.Module):
    """Convolution Block"""
    """"""
    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()

        self.conv = nn.Sequential(

            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True), #inplace为True, 将会改变输入的数据,否则不会改变原输入,只会产生新的输出

            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class upconv_block(nn.Module):
    """upConvolution Block"""

    def __init__(self, in_ch, out_ch):
        super(upconv_block, self).__init__()
        self.upconv = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.upconv(x)
        return x


class U_Net0(nn.Module):
    """U-Net"""
    """ """
    def __init__(self, in_ch=3, out_ch=1):  #out_ch 是输出类别还是输出的通道数 要考虑清楚
        super(U_Net0, self).__init__()

        # n1 = 64
        filters = [64, 128, 256, 512, 1024]
        # filters = [32, 64, 128, 256, 512]
        # filters = [16, 32, 64, 128, 256]
        # filters = [8, 16, 32, 64, 128]
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(in_ch, filters[0])
        self.Conv2 = conv_block(filters[0], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])
        self.Conv4 = conv_block(filters[2], filters[3])
        self.Conv5 = conv_block(filters[3], filters[4])

        self.Up5 = upconv_block(filters[4], filters[3])
        self.Up_conv5 = conv_block(filters[4], filters[3])

        self.Up4 = upconv_block(filters[3], filters[2])
        self.Up_conv4 = conv_block(filters[3], filters[2])

        self.Up3 = upconv_block(filters[2], filters[1])
        self.Up_conv3 = conv_block(filters[2], filters[1])

        self.Up2 = upconv_block(filters[1], filters[0])
        self.Up_conv2 = conv_block(filters[1], filters[0])

        self.Conv = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0) #为什么最后一个不用加bias
        """
        At the final layer a 1*1 convolution is used to map each 64-componet feature
        to the desired number of classes
        """

    def forward(self, x):

        """Contracting path"""  # e means encoder
        e1 = self.Conv1(x)

        e2 = self.Maxpool(e1)
        e2 = self.Conv2(e2)

        e3 = self.Maxpool(e2)
        e3 = self.Conv3(e3)

        e4 = self.Maxpool(e3)
        e4 = self.Conv4(e4)

        e5 = self.Maxpool(e4)
        e5 = self.Conv5(e5)

        """Expansive path"""  # d means decoder
        d5 = self.Up5(e5)
        d5 = torch.cat((e4, d5), dim=1)  # 按维数1拼接（横着拼）A,B = AB;维度0,竖着拼A,B= A在上,B在下
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((e3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((e1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        out = self.Conv(d2)

        return out


class U_Net4(nn.Module):
    """U-Net"""
    """ """
    def __init__(self, in_ch=3, out_ch=1):  #out_ch 是输出类别还是输出的通道数 要考虑清楚
        super(U_Net4, self).__init__()

        # n1 = 64
        # filters = [64, 128, 256, 512, 1024]
        # filters = [32, 64, 128, 256, 512]
        filters = [16, 32, 64, 128, 256]
        # filters = [8, 16, 32, 64, 128]
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(in_ch, filters[0])
        self.Conv2 = conv_block(filters[0], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])
        self.Conv4 = conv_block(filters[2], filters[3])
        self.Conv5 = conv_block(filters[3], filters[4])

        self.Up5 = upconv_block(filters[4], filters[3])
        self.Up_conv5 = conv_block(filters[4], filters[3])

        self.Up4 = upconv_block(filters[3], filters[2])
        self.Up_conv4 = conv_block(filters[3], filters[2])

        self.Up3 = upconv_block(filters[2], filters[1])
        self.Up_conv3 = conv_block(filters[2], filters[1])

        self.Up2 = upconv_block(filters[1], filters[0])
        self.Up_conv2 = conv_block(filters[1], filters[0])

        self.Conv = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0) #为什么最后一个不用加bias
        """
        At the final layer a 1*1 convolution is used to map each 64-componet feature
        to the desired number of classes
        """

    def forward(self, x):

        """Contracting path"""  # e means encoder
        e1 = self.Conv1(x)

        e2 = self.Maxpool(e1)
        e2 = self.Conv2(e2)

        e3 = self.Maxpool(e2)
        e3 = self.Conv3(e3)

        e4 = self.Maxpool(e3)
        e4 = self.Conv4(e4)

        e5 = self.Maxpool(e4)
        e5 = self.Conv5(e5)

        """Expansive path"""  # d means decoder
        d5 = self.Up5(e5)
        d5 = torch.cat((e4, d5), dim=1)  # 按维数1拼接（横着拼）A,B = AB;维度0,竖着拼A,B= A在上,B在下
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((e3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((e1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        out = self.Conv(d2)

        return out


class U_Net8(nn.Module):
    """U-Net"""
    """ """
    def __init__(self, in_ch=3, out_ch=1):  #out_ch 是输出类别还是输出的通道数 要考虑清楚
        super(U_Net8, self).__init__()

        # n1 = 64
        # filters = [64, 128, 256, 512, 1024]
        # filters = [32, 64, 128, 256, 512]
        # filters = [16, 32, 64, 128, 256]
        filters = [8, 16, 32, 64, 128]
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(in_ch, filters[0])
        self.Conv2 = conv_block(filters[0], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])
        self.Conv4 = conv_block(filters[2], filters[3])
        self.Conv5 = conv_block(filters[3], filters[4])

        self.Up5 = upconv_block(filters[4], filters[3])
        self.Up_conv5 = conv_block(filters[4], filters[3])

        self.Up4 = upconv_block(filters[3], filters[2])
        self.Up_conv4 = conv_block(filters[3], filters[2])

        self.Up3 = upconv_block(filters[2], filters[1])
        self.Up_conv3 = conv_block(filters[2], filters[1])

        self.Up2 = upconv_block(filters[1], filters[0])
        self.Up_conv2 = conv_block(filters[1], filters[0])

        self.Conv = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0) #为什么最后一个不用加bias
        """
        At the final layer a 1*1 convolution is used to map each 64-componet feature
        to the desired number of classes
        """

    def forward(self, x):

        """Contracting path"""  # e means encoder
        e1 = self.Conv1(x)

        e2 = self.Maxpool(e1)
        e2 = self.Conv2(e2)

        e3 = self.Maxpool(e2)
        e3 = self.Conv3(e3)

        e4 = self.Maxpool(e3)
        e4 = self.Conv4(e4)

        e5 = self.Maxpool(e4)
        e5 = self.Conv5(e5)

        """Expansive path"""  # d means decoder
        d5 = self.Up5(e5)
        d5 = torch.cat((e4, d5), dim=1)  # 按维数1拼接（横着拼）A,B = AB;维度0,竖着拼A,B= A在上,B在下
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((e3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((e1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        out = self.Conv(d2)

        return out


class U_Net(nn.Module):
    """U-Net"""
    """ """
    def __init__(self, in_ch=3, out_ch=1):  #out_ch 是输出类别还是输出的通道数 要考虑清楚
        super(U_Net, self).__init__()

        # n1 = 64
        # filters = [64, 128, 256, 512, 1024]
        filters = [32, 64, 128, 256, 512]
        # filters = [16, 32, 64, 128, 256]
        # filters = [8, 16, 32, 64, 128]
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(in_ch, filters[0])
        self.Conv2 = conv_block(filters[0], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])
        self.Conv4 = conv_block(filters[2], filters[3])
        self.Conv5 = conv_block(filters[3], filters[4])

        self.Up5 = upconv_block(filters[4], filters[3])
        self.Up_conv5 = conv_block(filters[4], filters[3])

        self.Up4 = upconv_block(filters[3], filters[2])
        self.Up_conv4 = conv_block(filters[3], filters[2])

        self.Up3 = upconv_block(filters[2], filters[1])
        self.Up_conv3 = conv_block(filters[2], filters[1])

        self.Up2 = upconv_block(filters[1], filters[0])
        self.Up_conv2 = conv_block(filters[1], filters[0])

        self.Conv = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0) #为什么最后一个不用加bias
        """
        At the final layer a 1*1 convolution is used to map each 64-componet feature
        to the desired number of classes
        """

    def forward(self, x):

        """Contracting path"""  # e means encoder
        e1 = self.Conv1(x)

        e2 = self.Maxpool(e1)
        e2 = self.Conv2(e2)

        e3 = self.Maxpool(e2)
        e3 = self.Conv3(e3)

        e4 = self.Maxpool(e3)
        e4 = self.Conv4(e4)

        e5 = self.Maxpool(e4)
        e5 = self.Conv5(e5)

        """Expansive path"""  # d means decoder
        d5 = self.Up5(e5)
        d5 = torch.cat((e4, d5), dim=1)  # 按维数1拼接（横着拼）A,B = AB;维度0,竖着拼A,B= A在上,B在下
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((e3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((e1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        out = self.Conv(d2)

        return out


class U_Net_Aspp(nn.Module):
    """U-Net"""
    """ """
    def __init__(self, in_ch=3, out_ch=1):  #out_ch 是输出类别还是输出的通道数 要考虑清楚
        super(U_Net_Aspp, self).__init__()

        # n1 = 64
        # filters = [64, 128, 256, 512, 1024]
        filters = [32, 64, 128, 256, 512]
        # filters = [16, 32, 64, 128, 256]
        # filters = [8, 16, 32, 64, 128]
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(in_ch, filters[0])
        self.Conv2 = conv_block(filters[0], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])
        self.Conv4 = conv_block(filters[2], filters[3])
        self.ASPP = ASPP(filters[3], filters[4])  # ASPP

        self.Up5 = upconv_block(filters[4], filters[3])
        self.Up_conv5 = conv_block(filters[4], filters[3])

        self.Up4 = upconv_block(filters[3], filters[2])
        self.Up_conv4 = conv_block(filters[3], filters[2])

        self.Up3 = upconv_block(filters[2], filters[1])
        self.Up_conv3 = conv_block(filters[2], filters[1])

        self.Up2 = upconv_block(filters[1], filters[0])
        self.Up_conv2 = conv_block(filters[1], filters[0])

        self.Conv = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0) #为什么最后一个不用加bias
        """
        At the final layer a 1*1 convolution is used to map each 64-componet feature
        to the desired number of classes
        """

    def forward(self, x):

        """Contracting path"""  # e means encoder
        e1 = self.Conv1(x)

        e2 = self.Maxpool(e1)
        e2 = self.Conv2(e2)

        e3 = self.Maxpool(e2)
        e3 = self.Conv3(e3)

        e4 = self.Maxpool(e3)
        e4 = self.Conv4(e4)

        e5 = self.Maxpool(e4)
        #e5 = self.Conv5(e5)

        ASPP = self.ASPP(e5)
        """Expansive path"""  # d means decoder
        d5 = self.Up5(ASPP)
        d5 = torch.cat((e4, d5), dim=1)  # 按维数1拼接（横着拼）A,B = AB;维度0,竖着拼A,B= A在上,B在下
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((e3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((e1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        out = self.Conv(d2)

        return out


class U_Net_ASPP_ALL(nn.Module):
    """U-Net"""
    """ """
    def __init__(self, in_ch=3, out_ch=1):  #out_ch 是输出类别还是输出的通道数 要考虑清楚
        super(U_Net_ASPP_ALL, self).__init__()

        # n1 = 64
        # filters = [64, 128, 256, 512, 1024]
        filters = [32, 64, 128, 256, 512]
        # filters = [16, 32, 64, 128, 256]
        # filters = [8, 16, 32, 64, 128]
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(in_ch, filters[0])
        self.Conv2 = conv_block(filters[0], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])
        self.Conv4 = conv_block(filters[2], filters[3])
        self.Conv5 = ASPP2(filters[3], filters[4])  # ASPP
        self.skip_aspp1 = ASPP2(filters[0], filters[0], rate=[8, 16, 24])
        self.skip_aspp2 = ASPP2(filters[1], filters[1], rate=[6, 12, 18])
        self.skip_aspp3 = ASPP2(filters[2], filters[2], rate=[4, 8, 12])
        self.skip_aspp4 = ASPP2(filters[3], filters[3], rate=[2, 4, 6])
        self.skip_aspp5 = ASPP2(filters[4], filters[4], rate=[1, 2, 4])

        self.Up5 = upconv_block(filters[4], filters[3])
        self.Up_conv5 = conv_block(filters[4], filters[3])

        self.Up4 = upconv_block(filters[3], filters[2])
        self.Up_conv4 = conv_block(filters[3], filters[2])

        self.Up3 = upconv_block(filters[2], filters[1])
        self.Up_conv3 = conv_block(filters[2], filters[1])

        self.Up2 = upconv_block(filters[1], filters[0])
        self.Up_conv2 = conv_block(filters[1], filters[0])

        self.Conv = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0) #为什么最后一个不用加bias
        """
        At the final layer a 1*1 convolution is used to map each 64-componet feature
        to the desired number of classes
        """

    def forward(self, x):

        """Contracting path"""  # e means encoder
        e1 = self.Conv1(x)

        e2 = self.Maxpool(e1)
        e2 = self.Conv2(e2)

        e3 = self.Maxpool(e2)
        e3 = self.Conv3(e3)

        e4 = self.Maxpool(e3)
        e4 = self.Conv4(e4)

        e5 = self.Maxpool(e4)
        e5 = self.Conv5(e5)

        e5 = self.skip_aspp5(e5)
        """Expansive path"""  # d means decoder
        d5 = self.Up5(e5)
        d5 = torch.cat((self.skip_aspp4(e4), d5), dim=1)  # 按维数1拼接（横着拼）A,B = AB;维度0,竖着拼A,B= A在上,B在下
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((self.skip_aspp3(e3), d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((self.skip_aspp2(e2), d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((self.skip_aspp1(e1), d2), dim=1)
        d2 = self.Up_conv2(d2)

        out = self.Conv(d2)

        return out


class U_Net_Aspp2(nn.Module):
    """U-Net"""
    """ """
    def __init__(self, in_ch=3, out_ch=1):  #out_ch 是输出类别还是输出的通道数 要考虑清楚
        super(U_Net_Aspp2, self).__init__()

        # n1 = 64
        # filters = [64, 128, 256, 512, 1024]
        filters = [32, 64, 128, 256, 512]
        # filters = [16, 32, 64, 128, 256]
        # filters = [8, 16, 32, 64, 128]
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(in_ch, filters[0])
        self.Conv2 = conv_block(filters[0], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])
        self.Conv4 = conv_block(filters[2], filters[3])
        self.Conv5 = conv_block(filters[3], filters[4])
        self.ASPP = ASPP(filters[2], filters[3])  # ASPP

        self.Up5 = upconv_block(filters[4], filters[3])
        self.Up_conv5 = conv_block(filters[4], filters[3])

        self.Up4 = upconv_block(filters[3], filters[2])
        self.Up_conv4 = conv_block(filters[3], filters[2])

        self.Up3 = upconv_block(filters[2], filters[1])
        self.Up_conv3 = conv_block(filters[2], filters[1])

        self.Up2 = upconv_block(filters[1], filters[0])
        self.Up_conv2 = conv_block(filters[1], filters[0])

        self.Conv = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0) #为什么最后一个不用加bias
        """
        At the final layer a 1*1 convolution is used to map each 64-componet feature
        to the desired number of classes
        """

    def forward(self, x):

        """Contracting path"""  # e means encoder
        e1 = self.Conv1(x)

        e2 = self.Maxpool(e1)
        e2 = self.Conv2(e2)

        e3 = self.Maxpool(e2)
        e3 = self.Conv3(e3)

        e4 = self.Maxpool(e3)
        e4 = self.Conv4(e4)

        e5 = self.Maxpool(e4)
        e5 = self.Conv5(e5)


        """Expansive path"""  # d means decoder
        d5 = self.Up5(e5)
        ASPP = self.ASPP(e4)
        d5 = torch.cat((ASPP, d5), dim=1)  # 按维数1拼接（横着拼）A,B = AB;维度0,竖着拼A,B= A在上,B在下
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((e3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((e1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        out = self.Conv(d2)

        return out


class U_Net_Att(nn.Module):
    """U-Net"""
    """ """
    def __init__(self, in_ch=3, out_ch=1):  #out_ch 是输出类别还是输出的通道数 要考虑清楚
        super(U_Net_Att, self).__init__()

        # n1 = 64
        # filters = [64, 128, 256, 512, 1024]
        filters = [32, 64, 128, 256, 512]
        # filters = [16, 32, 64, 128, 256]
        # filters = [8, 16, 32, 64, 128]
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(in_ch, filters[0])
        self.Conv2 = conv_block(filters[0], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])
        self.Conv4 = conv_block(filters[2], filters[3])
        self.Conv5 = conv_block(filters[3], filters[4])  # ASPP

        self.attn1 = AttentionBlock(filters[2], filters[4], filters[4])
        self.Up5 = upconv_block(filters[4], filters[3])
        self.Up_conv5 = conv_block(filters[4], filters[3])

        self.attn2 = AttentionBlock(filters[1], filters[3], filters[3])
        self.Up4 = upconv_block(filters[3], filters[2])
        self.Up_conv4 = conv_block(filters[3], filters[2])

        self.attn3 = AttentionBlock(filters[0], filters[2], filters[2])
        self.Up3 = upconv_block(filters[2], filters[1])
        self.Up_conv3 = conv_block(filters[2], filters[1])

        self.Up2 = upconv_block(filters[1], filters[0])
        self.Up_conv2 = conv_block(filters[1], filters[0])

        self.Conv = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0) #为什么最后一个不用加bias
        """
        At the final layer a 1*1 convolution is used to map each 64-componet feature
        to the desired number of classes
        """

    def forward(self, x):

        """Contracting path"""  # e means encoder
        e1 = self.Conv1(x)

        e2 = self.Maxpool(e1)
        e2 = self.Conv2(e2)

        e3 = self.Maxpool(e2)
        e3 = self.Conv3(e3)

        e4 = self.Maxpool(e3)
        e4 = self.Conv4(e4)

        e5 = self.Maxpool(e4)
        #e5 = self.Conv5(e5)

        ASPP = self.ASPP(e5)
        """Expansive path"""  # d means decoder
        d5 = self.Up5(ASPP)
        d5 = torch.cat((e4, d5), dim=1)  # 按维数1拼接（横着拼）A,B = AB;维度0,竖着拼A,B= A在上,B在下
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((e3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((e1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        out = self.Conv(d2)

        return out



class conv_block_nested(nn.Module):

    def __init__(self, in_ch, mid_ch, out_ch):
        super(conv_block_nested, self).__init__()
        self.activation = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(mid_ch)
        self.conv2 = nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)

        x = self.conv2(x)
        x = self.bn2(x)
        output = self.activation(x)

        return output

class NestedUNet(nn.Module):
    """
    Implementation of this paper:
    https://arxiv.org/pdf/1807.10165.pdf
    """

    def __init__(self, in_ch=3, out_ch=1):
        super(NestedUNet, self).__init__()

        n1 = 32
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = conv_block_nested(in_ch, filters[0], filters[0])
        self.conv1_0 = conv_block_nested(filters[0], filters[1], filters[1])
        self.conv2_0 = conv_block_nested(filters[1], filters[2], filters[2])
        self.conv3_0 = conv_block_nested(filters[2], filters[3], filters[3])
        self.conv4_0 = conv_block_nested(filters[3], filters[4], filters[4])

        self.conv0_1 = conv_block_nested(filters[0] + filters[1], filters[0], filters[0])
        self.conv1_1 = conv_block_nested(filters[1] + filters[2], filters[1], filters[1])
        self.conv2_1 = conv_block_nested(filters[2] + filters[3], filters[2], filters[2])
        self.conv3_1 = conv_block_nested(filters[3] + filters[4], filters[3], filters[3])

        self.conv0_2 = conv_block_nested(filters[0] * 2 + filters[1], filters[0], filters[0])
        self.conv1_2 = conv_block_nested(filters[1] * 2 + filters[2], filters[1], filters[1])
        self.conv2_2 = conv_block_nested(filters[2] * 2 + filters[3], filters[2], filters[2])

        self.conv0_3 = conv_block_nested(filters[0] * 3 + filters[1], filters[0], filters[0])
        self.conv1_3 = conv_block_nested(filters[1] * 3 + filters[2], filters[1], filters[1])

        self.conv0_4 = conv_block_nested(filters[0] * 4 + filters[1], filters[0], filters[0])

        self.final = nn.Conv2d(filters[0], out_ch, kernel_size=1)

    def forward(self, x):
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.Up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.Up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.Up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.Up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.Up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.Up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.Up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.Up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.Up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.Up(x1_3)], 1))

        output = self.final(x0_4)
        return output
