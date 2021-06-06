import torch
import math
import torch.nn as nn
import torch.nn.functional as F

class se_res_extracter(nn.Module):
    def __init__(self, in_num=1, out_num=1, filters=[16,32,64,96,128],train=True):
        super(se_res_extracter, self).__init__()
        self.filters = filters 
        self.layer_num = len(filters) # 5
        self.aniso_num = 3 # the number of anisotropic conv layers
        self.training = train
        self.qua = quality(filters[0])
        self.downC = nn.ModuleList(
                  [res_unet_AnisoBlock(in_num, filters[0])]
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
        self.upScore = nn.Sequential(
            res_unet_AnisoBlock(filters[-1],filters[0]),
            nn.Conv2d(filters[0], filters[0], kernel_size=(3,3), stride=1, padding=(1,1), bias=True)
                       )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0) 

    def forward(self, x, qua = False):
        down_u = [None]*(self.layer_num-1)
        for i in range(self.layer_num-1):
            down_u[i] = self.downC[i](x)
            x = self.downS[i](down_u[i])

        x = self.center(x)
        score = self.upScore(x)
        score = self.qua(score)
        return score        

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
        self.se = se_module(channel = out_planes)

    def forward(self, x):
        residual  = self.block1(x)
        out = self.se(residual) + self.block2(residual)
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
        out = self.se(residual) + self.block2(residual)
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

class quality(nn.Module):
    def __init__(self, channel):  
        super(quality,self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                    nn.Linear(channel,channel),
                    nn.ReLU(inplace=True),
                    nn.Linear(channel, 1),
                    #nn.Softsign(),
                    nn.Sigmoid()
                    #nn.LogSigmoid(),
                )
    def forward(self,x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b)*2
        return F.softmax(y,dim=0).view(b,1,1)
