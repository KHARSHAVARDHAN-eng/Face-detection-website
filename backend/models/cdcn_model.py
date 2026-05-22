import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv2d_cd(nn.Module):
    """
    Central Difference Convolutional Layer (CDC)
    Formula:
      y(p_0) = w(p_n) * x(p_0 + p_n) - theta * x(p_0) * sum(w(p_n))
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=0.7):
        super(Conv2d_cd, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation,
                              groups=groups, bias=bias)
        self.theta = theta

    def forward(self, x):
        # Regular standard convolution
        out_normal = self.conv(x)

        if math.isclose(self.theta, 0.0):
            return out_normal

        # Sum kernel weights along spatial dimensions (H, W) -> shape: [out_channels, in_channels, 1, 1]
        kernel_diff = self.conv.weight.sum(dim=(2, 3), keepdim=True)
        
        # Convolve with sum of weights to get center pixel contribution
        # Stride remains the same, padding is 0 (as kernel is 1x1)
        out_diff = F.conv2d(input=x, weight=kernel_diff, bias=None,
                            stride=self.conv.stride, padding=0,
                            groups=self.conv.groups)

        # Output = Conv(x) - theta * Center_Conv(x)
        return out_normal - self.theta * out_diff


class CDCN(nn.Module):
    """
    Central Difference Convolutional Network (CDCN) for Face Anti-Spoofing.
    Input size: [N, 3, 256, 256]
    Output size: [N, 1, 32, 32] (Estimated depth map)
    """
    def __init__(self, basic_conv=Conv2d_cd, theta=0.7):
        super(CDCN, self).__init__()
        self.conv1 = basic_conv(3, 64, kernel_size=3, stride=1, padding=1, theta=theta)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        
        self.block1 = nn.Sequential(
            basic_conv(64, 128, kernel_size=3, stride=1, padding=1, theta=theta),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.block2 = nn.Sequential(
            basic_conv(128, 128, kernel_size=3, stride=1, padding=1, theta=theta),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.block3 = nn.Sequential(
            basic_conv(128, 128, kernel_size=3, stride=1, padding=1, theta=theta),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Depth estimator conv to output a 1-channel depth map
        self.depth_conv = nn.Conv2d(128, 1, kernel_size=3, stride=1, padding=1)
        
    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.block1(x)  # [N, 128, 128, 128]
        x = self.block2(x)  # [N, 128, 64, 64]
        x = self.block3(x)  # [N, 128, 32, 32]
        depth = self.depth_conv(x)  # [N, 1, 32, 32]
        return depth
