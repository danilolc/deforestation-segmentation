import torch
import torch.nn as nn

class ConvolutionBlock(nn.Module):
    def __init__(self, in_c, out_c, kernel_size, padding='same', stride=1, dilation=1, bias=True, residual=False):
        super().__init__()

        self.residual = residual

        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, 
                    kernel_size=kernel_size,
                    padding=padding,
                    padding_mode='replicate',
                    stride=stride, 
                    dilation=dilation,
                    bias=bias),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        if self.residual:
            return x + self.conv(x)
        return self.conv(x)
    
class TransposeConvBlock(nn.Module):
    def __init__(self, in_c, out_c, kernel_size, stride=2):
        super().__init__()

        self.conv = nn.Sequential(
            nn.ConvTranspose2d(in_c, out_c, 
                               kernel_size=kernel_size, 
                               stride=stride),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class ASPP(nn.Module):
    def __init__(self, in_c, out_c, residual=False):
        super().__init__()

        self.residual = residual

        self.conv1 = ConvolutionBlock(in_c, out_c, kernel_size=1)
        self.conv2 = ConvolutionBlock(in_c, out_c, kernel_size=3, dilation=6)  # 3
        self.conv3 = ConvolutionBlock(in_c, out_c, kernel_size=3, dilation=12) # 6
        self.conv4 = ConvolutionBlock(in_c, out_c, kernel_size=3, dilation=18) # 12

        self.conv5 = nn.Sequential(
            nn.AvgPool2d(8),
            ConvolutionBlock(in_c, out_c, kernel_size=1),
            nn.Upsample(scale_factor=8, mode='bilinear'),
        )

        self.conv6 = ConvolutionBlock(5 * out_c, out_c, kernel_size=1)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)
        x5 = self.conv5(x)

        x6 = torch.cat([x1, x2, x3, x4, x5], dim=1)
        x6 = self.conv6(x6)

        if self.residual:
            return x + x6
        
        return x6

# https://arxiv.org/pdf/1804.03999.pdf
class AdditiveAttentionGate(nn.Module):
    def __init__(self, fx, fg, fint):
        super().__init__()

        self.Wx = nn.Sequential(
            nn.Conv2d(fx, fint, kernel_size=1, stride=2),
            nn.BatchNorm2d(fint),
        )
        self.Wg = nn.Sequential(
            nn.Conv2d(fg, fint, kernel_size=1, bias=False),
            nn.BatchNorm2d(fint),
        )

        self.Psi = nn.Sequential(
            nn.Conv2d(fint, 1, kernel_size=1),
            nn.BatchNorm2d(1),
        )

    def forward(self, x, g):
        
        xv = self.Wx(x)
        gv = self.Wg(g)
        
        s = nn.functional.relu(xv + gv, inplace=True)
        s = self.Psi(s)

        s = torch.sigmoid(s)
        s = nn.functional.interpolate(s,
                                      scale_factor=2,
                                      mode='bilinear',
                                      align_corners=True)

        return s

class Unet(nn.Module):
    def __init__(self, in_c, out_c, attention=False):
        super().__init__()

        self.downsample = nn.MaxPool2d(2)

        self.attention = attention
        if self.attention:
            self.att1 = AdditiveAttentionGate(32, 64, 64 * 2) #8
            self.att2 = AdditiveAttentionGate(16, 32, 32 * 2) #4

        self.conv1 = nn.Sequential(
            ConvolutionBlock(in_c, 16, kernel_size=3),
            ASPP(16, 16, residual=True),
            ASPP(16, 16, residual=True),
            ASPP(16, 16, residual=True),
        )
        self.conv2 = nn.Sequential(
            ConvolutionBlock(16, 32, kernel_size=3),
            ASPP(32, 32, residual=True),
            ASPP(32, 32, residual=True),
            ASPP(32, 32, residual=True),
        )
        self.conv3 = nn.Sequential(
            ConvolutionBlock(32, 64, kernel_size=3),
            ConvolutionBlock(64, 64, kernel_size=3),
        )

        self.tconv1 = TransposeConvBlock(64, 32, kernel_size=2)

        self.conv4 = nn.Sequential(
            ConvolutionBlock(64, 32, kernel_size=3),
            
            ASPP(32, 32, residual=True),
            ASPP(32, 32, residual=True),
            ASPP(32, 32, residual=True),
        )

        self.tconv2 = TransposeConvBlock(32, 16, kernel_size=2)

        self.conv5 = nn.Sequential(
            ConvolutionBlock(32, 16, kernel_size=3),

            ASPP(16, 16, residual=True),
            ASPP(16, 16, residual=True),
            ASPP(16, 16, residual=True),
            ConvolutionBlock(16, out_c, kernel_size=3),
        )

    def forward(self, x):

        x = self.conv1(x)

        x1 = self.downsample(x)
        x1 = self.conv2(x1)

        x2 = self.downsample(x1)
        x2 = self.conv3(x2)

        if (self.attention):
            att = self.att1(x1, x2)
            x2 = self.tconv1(x2)
            x2 = torch.cat([x1 * att, x2], dim=1)
        else:
            x2 = self.tconv1(x2)
            x2 = torch.cat([x1, x2], dim=1)
        
        x2 = self.conv4(x2)

        if (self.attention):
            att = self.att2(x, x2)
            x2 = self.tconv2(x2)
            x2 = torch.cat([x * att, x2], dim=1)
        else:
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