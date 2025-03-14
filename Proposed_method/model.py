import torch
import torch.nn as nn
import numpy as np
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def conv3x3_real(in_channels, out_channels, stride=1, padding=(1,1)):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=padding,
        bias=False
    )

def conv1x1_real(in_channels, out_channels, stride=1, padding=0):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=1,
        stride=stride,
        padding=padding,
        bias=False
    )

class BottleNeck_real(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels_mid,
                 out_channels
                 ):
        super(BottleNeck_real, self).__init__()
        self.cov1 = conv1x1_real(in_channels=in_channels, out_channels=out_channels_mid)
        self.bn1 = nn.BatchNorm2d(out_channels_mid)
        self.tanh = nn.Tanh()
        self.cov2 = conv3x3_real(in_channels=out_channels_mid, out_channels=out_channels_mid)
        self.bn2 = nn.BatchNorm2d(out_channels_mid)
        self.cov3 = conv1x1_real(in_channels=out_channels_mid, out_channels=out_channels)
        self.bn3 = nn.BatchNorm2d(out_channels)


        self.short_cut = conv1x1_real(in_channels=in_channels, out_channels=out_channels)
        self.short_cut_bn = nn.BatchNorm2d(out_channels)

    def forward(self,x):
        identity = x
        out = self.cov1(x)
        out = self.bn1(out)
        out = self.tanh(out)

        out = self.cov2(out)
        out = self.bn2(out)
        out = self.tanh(out)

        out = self.cov3(out)
        out = self.bn3(out)

        identity = self.short_cut(identity)
        identity = self.short_cut_bn(identity)

        out = out + identity
        out = self.tanh(out)

        return out


#######################################
##         Proposed_model          ##
#######################################
class Proposed_model(nn.Module):
    def __init__(self, D):
        super(Proposed_model, self).__init__()
        self.cov1 = conv3x3_real(in_channels=2, out_channels=64)

        self.block1_1 = BottleNeck_real(in_channels=64, out_channels_mid=64, out_channels=64)

        self.block1_2 = BottleNeck_real(in_channels=64, out_channels_mid=128, out_channels=128)
        self.block1_3 = BottleNeck_real(in_channels=128, out_channels_mid=256, out_channels=256)
        self.block1_4 = BottleNeck_real(in_channels=256, out_channels_mid=512, out_channels=512)

        if D == 1:
            self.cov2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 2), stride=1, padding=(1, 0), bias=False)  # (8,7)
        elif D == 2:
            self.cov2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=1, padding=(1, 0), bias=False)  # (8,6)
        elif D == 3:
            self.cov2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 4), stride=1, padding=(1, 0), bias=False)    # (8,5)

        self.cov3 = conv1x1_real(in_channels=512, out_channels=2)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.cov1(x)

        x = self.block1_1(x)
        x = self.block1_2(x)
        x = self.block1_3(x)
        x = self.block1_4(x)

        x = self.cov2(x)

        x = self.cov3(x)
        return x
