""" Parts of the U-Net model This is the network_structure module for the Sea Ice Extent GTC Project.

This module contains classes that define the U-Net structure as defined in the original U-Net paper
and repository (https://github.com/milesial/Pytorch-UNet) which was accessed in January 2022."""

from unet.shared import *


class DoubleConv(nn.Module):
    """(convolution => BatchNorm => ReLU) * 2"""

    def __init__(cls, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        cls.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(cls, x):
        return cls.double_conv(x)


class Down(nn.Module):
    """Down pass of the U-Net - Downscaling with maxpool then double conv"""

    def __init__(cls, in_channels, out_channels):
        super().__init__()
        cls.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(cls, x):
        return cls.maxpool_conv(x)


class Up(nn.Module):
    """Up pass of the U-Net - upscaling then double conv"""

    def __init__(cls, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            cls.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            cls.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            cls.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            cls.conv = DoubleConv(in_channels, out_channels)

    def forward(cls, x1, x2):
        x1 = cls.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return cls.conv(x)


class OutConv(nn.Module):
    """ Out convolution at the end of the U-Net that defines the output.
    """
    def __init__(cls, in_channels, out_channels):
        super(OutConv, cls).__init__()
        cls.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(cls, x):
        return cls.conv(x)


class UNet(nn.Module):
    """ Full assemply of the U-Net parts to form the complete network.
    """
    def __init__(cls, n_channels, n_classes, bilinear=True):
        super(UNet, cls).__init__()
        cls.n_channels = n_channels
        cls.n_classes = n_classes
        cls.bilinear = bilinear

        cls.inc = DoubleConv(n_channels, 64)
        cls.down1 = Down(64, 128)
        cls.down2 = Down(128, 256)
        cls.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        cls.down4 = Down(512, 1024 // factor)
        cls.up1 = Up(1024, 512 // factor, bilinear)
        cls.up2 = Up(512, 256 // factor, bilinear)
        cls.up3 = Up(256, 128 // factor, bilinear)
        cls.up4 = Up(128, 64, bilinear)
        cls.outc = OutConv(64, n_classes)

    def forward(cls, x):
        x1 = cls.inc(x)
        x2 = cls.down1(x1)
        x3 = cls.down2(x2)
        x4 = cls.down3(x3)
        x5 = cls.down4(x4)
        x = cls.up1(x5, x4)
        x = cls.up2(x, x3)
        x = cls.up3(x, x2)
        x = cls.up4(x, x1)
        logits = cls.outc(x)
        return logits
