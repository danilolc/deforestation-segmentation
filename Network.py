import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvolutionBlock(nn.Module):
    def __init__(self, in_c, out_c, residual=False, bias=False):
        super().__init__()
        self.residual = residual
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding='same', bias=bias),
            nn.GroupNorm(1, out_c),
            nn.GELU(),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding='same', bias=bias),
            nn.GroupNorm(1, out_c),
        )

    def forward(self, x):
        if self.residual:
            return F.gelu(x + self.conv(x))
        return F.gelu(self.conv(x))

class TransposeConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.conv = nn.Sequential(
            nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2),
            nn.GroupNorm(1, out_c),
            nn.GELU(),
        )

    def forward(self, x):
        return self.conv(x)

class ASPP(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=1)
        self.conv2 = nn.Conv2d(in_c, out_c, kernel_size=3, padding='same', dilation=6)  # 3
        self.conv3 = nn.Conv2d(in_c, out_c, kernel_size=3, padding='same', dilation=12) # 6
        self.conv4 = nn.Conv2d(in_c, out_c, kernel_size=3, padding='same', dilation=18) # 12

        self.conv5 = nn.Sequential(
            nn.AvgPool2d(8),
            nn.Conv2d(in_c, out_c, kernel_size=1),
            nn.Upsample(scale_factor=8, mode='bilinear'),
        )

        self.conv6 = nn.Conv2d(5 * out_c, out_c, kernel_size=1)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)
        x5 = self.conv5(x)

        x6 = torch.cat([x1, x2, x3, x4, x5], dim=1)
        x6 = self.conv6(x6)

        return x6

class Unet(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.conv1 = nn.Sequential(
            ConvolutionBlock(in_c, 16, bias=True),
            ConvolutionBlock(16, 16, residual=True),
            ConvolutionBlock(16, 16, residual=True),
        )
        self.conv2 = nn.Sequential(
            nn.MaxPool2d(2),
            ConvolutionBlock(16, 32),
            ConvolutionBlock(32, 32, residual=True),
            ConvolutionBlock(32, 32, residual=True),
        )
        self.conv3 = nn.Sequential(
            nn.MaxPool2d(2),
            ConvolutionBlock(32, 64),
            ConvolutionBlock(64, 64, residual=True),
            ConvolutionBlock(64, 64, residual=True),
        )

        self.tconv1 = TransposeConvBlock(64, 32)
        self.conv4 = nn.Sequential(
            ConvolutionBlock(64, 32),
            ConvolutionBlock(32, 32, residual=True),
            ConvolutionBlock(32, 32, residual=True),
        )

        self.tconv2 = TransposeConvBlock(32, 16)
        self.conv5 = nn.Sequential(
            ConvolutionBlock(32, 16),
            ConvolutionBlock(16, 16, residual=True),
            ConvolutionBlock(16, 16, residual=True),

            #ConvolutionBlock(16, out_c, bias=True),
        )

    def forward(self, x):

        x = self.conv1(x)
        x1 = self.conv2(x)
        x2 = self.conv3(x1)

        x2 = self.tconv1(x2)
        x2 = torch.cat([x1, x2], dim=1)
        x2 = self.conv4(x2)

        x2 = self.tconv2(x2)
        x2 = torch.cat([x, x2], dim=1)
        x2 = self.conv5(x2)
        
        return x2

#https://www.mdpi.com/2072-4292/11/9/1015/pdf
#@torch.compile
def ssim_loss(pred, target):
    mean_pred = torch.mean(pred, dim=(1, 2, 3))
    mean_target = torch.mean(target, dim=(1, 2, 3))

    var_pred = torch.var(pred, dim=(1, 2, 3))
    var_target = torch.var(target, dim=(1, 2, 3))

    cov = torch.mean(pred * target, dim=(1, 2, 3)) - mean_pred * mean_target

    c1 = 0.01 ** 2
    c2 = 0.03 ** 2

    ssim = (2 * mean_pred * mean_target + c1) * (2 * cov + c2) 
    ssim /= (mean_pred ** 2 + mean_target ** 2 + c1) * (var_pred + var_target + c2)

    return (1 - ssim) / 2