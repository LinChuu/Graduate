# sub-parts of the U-Net model

import torch
import torch.nn as nn
import torch.nn.functional as F

class double_conv(nn.Module):
    # Basic residual module of unet
    def __init__(self, in_planes, out_planes):
        super(double_conv, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_planes,  out_planes, kernel_size=(3,3), stride=1, padding=(1,1), bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True))
        self.block2 = nn.Sequential(
            nn.Conv2d(out_planes, out_planes, kernel_size=(3,3), stride=1, padding=(1,1), bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_planes, out_planes, kernel_size=(3,3), stride=1, padding=(1,1), bias=False),
            nn.BatchNorm2d(out_planes))
        self.block3 = nn.ReLU(inplace=True)
        self.se = se_module(channel = out_planes)

    def forward(self, x):
        residual  = self.block1(x)
        out = self.block2(residual) + self.se(residual)
        out = self.block3(out)
        return out    

class se_module(nn.Module):
    def __init__(self, channel, reduction=4):
        super(se_module, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // reduction),
                nn.ReLU(inplace=True),
                nn.Linear(channel // reduction, channel),
                nn.Sigmoid()
                )
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y
'''
class double_conv(nn.Module):
   
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

'''
class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(up, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch//2, in_ch//2, 2, stride=2)

        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX//2,
                        diffY // 2, diffY - diffY//2))
        
        # for padding issues, see 
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x
    
class downcat(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(downcat, self).__init__()
        self.downcat = nn.MaxPool2d(2)
    
        self.conv = double_conv(in_ch, out_ch)
        
    def forward(self,x1,x2):
        #print(x1.shape)
        x1 = self.downcat(x1)
        #print(x1.shape)
         # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX//2,
                        diffY // 2, diffY - diffY//2))
        x = torch.cat([x2, x1], dim=1)
        #print(x.shape)
        x = self.conv(x)
        return x
        
        
class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x
