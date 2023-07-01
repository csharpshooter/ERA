import torch.nn as nn
import torch.nn.functional as F
from .depthwise_seperable_conv2d import DepthwiseSeparableConv2d


class CNN_Model(nn.Module):

    def __init__(self):
        super(CNN_Model, self).__init__()

        self.inputblock = nn.Sequential(
            # Defining a 2D convolution layer                                               RF = 1
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, bias=False, dilation=1, padding=1),
            # RF = 3
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )

        self.convblock1 = nn.Sequential(
            # Defining a 2D convolution layer
            # nn.Conv2d(32, 64, kernel_size=3, stride=1, bias=False, dilation=1, padding=1),  # RF = 5
            DepthwiseSeparableConv2d(32, 32, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, dilation=2, bias=False, padding=1),  # RF = 7
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )

        self.convblock2 = nn.Sequential(
            # Defining a 2D convolution layer
            nn.Conv2d(64, 32, kernel_size=3, stride=1, bias=False, dilation=1, padding=1),  # RF = 11
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            # nn.Conv2d(32, 64, kernel_size=3, stride=1, dilation=1, bias=False, padding=1),  # RF = 15
            DepthwiseSeparableConv2d(32, 32, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, dilation=2, bias=False, padding=2), # RF = 19
            # RF = 26
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )

        self.convblock3 = nn.Sequential(
            # Defining a 2D convolution layer
            nn.Conv2d(64, 32, kernel_size=3, stride=1, bias=False, dilation=1, padding=1),  # RF = 27
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            # nn.Conv2d(32, 64, kernel_size=3, stride=1, dilation=1, bias=False, padding=1),  # RF = 35
            DepthwiseSeparableConv2d(32, 32, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, dilation=2, bias=False, padding=2),# RF = 43
            # RF = 26
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )

        self.convblock4 = nn.Sequential(
            # Defining a 2D convolution layer
            nn.Conv2d(64, 32, kernel_size=3, stride=1, bias=False, dilation=1, padding=1),  # RF = 59
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            # nn.Conv2d(32, 64, kernel_size=3, stride=1, dilation=1, bias=False, padding=1),  # RF = 75
            DepthwiseSeparableConv2d(32, 32, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, dilation=2, bias=False, padding=1),  # RF = 91
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )

        self.gap = nn.Sequential(nn.AvgPool2d(kernel_size=2))  # RF = 91

        self.linear = nn.Linear(64, 10, bias=False)  # RF = 91

    def forward(self, x):
        x = self.inputblock(x) # RF = 1
        x = self.convblock1(x) # RF = 7
        x = self.convblock2(x) # RF = 19
        x = self.convblock3(x) # RF = 43
        x = self.convblock4(x) # RF = 91
        x = self.gap(x) # RF = 91
        x = x.view(-1, 64) # RF = 91
        x = self.linear(x) # RF = 91
        return F.log_softmax(x, dim=-1)
