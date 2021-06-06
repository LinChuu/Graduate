import torch
import math
import torch.nn as nn
import torch.nn.functional as F

class seg_module(nn.Module):
    def __init__(self, n_channels=1, n_classes=1, filters=[16,32,64,96,128],train=True):
        super(seg_module, self).__init__()
        self.filters = filters 
        self.layer_num = len(filters) # 5
        self.aniso_num = 3 # the number of anisotropic conv layers
        self.training = train
        self.downC = nn.ModuleList(
                  [res_unet_AnisoBlock(n_channels, filters[0])]
                + [res_unet_AnisoBlock(filters[x], filters[x+1])
                      for x in range(self.aniso_num-1)] 
                + [res_unet_IsoBlock(filters[x], filters[x+1])
                      for x in range(self.aniso_num-1, self.layer_num-2)]) 

        self.downS = nn.ModuleList(
                [nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
                    for x in range(self.aniso_num)]
              + [nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
                    for x in range(self.aniso_num, self.layer_num-1)])

        self.center = res_unet_IsoBlock(filters[-2], filters[-1])

        self.upS = nn.ModuleList(
            [nn.Sequential(
                nn.Upsample(scale_factor=(2,2), mode='bilinear', align_corners=False),
                nn.Conv2d(filters[self.layer_num-1-x], filters[self.layer_num-2-x], kernel_size=(3,3), stride=1, padding=(1,1), bias=True))
                for x in range(self.layer_num-self.aniso_num-1)]
          + [nn.Sequential(
                nn.Upsample(scale_factor=(2,2), mode='bilinear', align_corners=False),
                nn.Conv2d(filters[self.layer_num-1-x], filters[self.layer_num-2-x], kernel_size=(3,3), stride=1, padding=(1,1), bias=True))
                for x in range(1, self.aniso_num+1)])

        self.upC = nn.ModuleList(
            [res_unet_IsoBlock(filters[self.layer_num-2-x], filters[self.layer_num-2-x])
                for x in range(self.layer_num-self.aniso_num-1)]
          + [res_unet_AnisoBlock(filters[self.layer_num-2-x], filters[self.layer_num-2-x])
                for x in range(1, self.aniso_num)]
          + [nn.Sequential(
                  res_unet_AnisoBlock(filters[0], filters[0]),
                  nn.Conv2d(filters[0], n_classes, kernel_size=(3,3), stride=1, padding=(1,1), bias=True))])
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0) 

    def forward(self, x):
        down_u = [None]*(self.layer_num-1)
        for i in range(self.layer_num-1):
            down_u[i] = self.downC[i](x)
            x = self.downS[i](down_u[i])

        x = self.center(x)
        for i in range(self.layer_num-2):
            x = down_u[self.layer_num-2-i] + self.upS[i](x)
            x = F.relu(x)
            x = self.upC[i](x)
            x = F.sigmoid(x)
        x = down_u[0] + self.upS[self.layer_num-2](x)
        x = F.relu(x)
        heatmap = self.upC[self.layer_num-2](x)
        return heatmap 

class res_unet_IsoBlock(nn.Module):
    # Basic residual module of unet
    def __init__(self, in_planes, out_planes):
        super(res_unet_IsoBlock, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_planes,  out_planes, kernel_size=(3,3), stride=1, padding=(1,1), bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ELU(alpha=1, inplace=True)
            )
        self.block2 = nn.Sequential(
            nn.Conv2d(out_planes, out_planes, kernel_size=(3,3), stride=1, padding=(1,1), bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ELU(alpha=1, inplace=True),            
            nn.Conv2d(out_planes, out_planes, kernel_size=(3,3), stride=1, padding=(1,1), bias=False),
            nn.BatchNorm2d(out_planes))
        self.block3 = nn.ELU(alpha=1, inplace=True)
        self.se = se_module(channel=out_planes)

    def forward(self, x):
        residual  = self.block1(x)
        out = self.block2(residual) + self.se(residual)
        out = self.block3(out)
        return out 

class res_unet_AnisoBlock(nn.Module):
    # Basic residual module of unet
    def __init__(self, in_planes, out_planes):
        super(res_unet_AnisoBlock, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_planes,  out_planes, kernel_size=(3,3), stride=1, padding=(1,1), bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ELU(alpha=1, inplace=True))
        self.block2 = nn.Sequential(
            nn.Conv2d(out_planes, out_planes, kernel_size=(3,3), stride=1, padding=(1,1), bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ELU(alpha=1, inplace=True),
            nn.Conv2d(out_planes, out_planes, kernel_size=(3,3), stride=1, padding=(1,1), bias=False),
            nn.BatchNorm2d(out_planes))
        self.block3 = nn.ELU(alpha=1, inplace=True)   
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
