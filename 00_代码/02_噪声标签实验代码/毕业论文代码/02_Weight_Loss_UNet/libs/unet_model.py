# full assembly of the sub-parts to form the complete net

import torch.nn.functional as F

import torch.nn as nn

import os



from libs.unet_parts import inconv,down,up,outconv


class UNet(nn.Module):
    def __init__(self,  n_classes,n_channels=3):
        super(UNet, self).__init__()
        filters = [32,64,96,128,256]  #[64,128,256,512,1024]
        self.n_classes = n_classes
        self.inc = inconv(n_channels, filters[0]) 
        self.down1 = down(filters[0], filters[1])
        self.down2 = down(filters[1], filters[2])
        self.down3 = down(filters[2], filters[3])
        self.down4 = down(filters[3], filters[3])
        self.up1 = up(filters[3]+filters[3], filters[2])
        self.up2 = up(filters[2]+filters[2], filters[1])
        self.up3 = up(filters[1]+filters[1], filters[0])
        self.up4 = up(filters[0]+filters[0], 16)
        self.outc = outconv(16, n_classes)
        
        

    def forward(self, x):
        x_size = x.size()
        x1 = self.inc(x)
           
        x2 = self.down1(x1)
            
        x3 = self.down2(x2)
             
        x4 = self.down3(x3)
          
        x5 = self.down4(x4)
         
        x = self.up1(x5, x4)
         
        x = self.up2(x, x3)
           
        x = self.up3(x, x2)
         
        
        x = self.up4(x, x1)
 
        x = self.outc(x)
          
        x = F.interpolate(x, x_size[2:], mode='bilinear', align_corners=False)
        return x
        '''
        if labels is not None:

            classwise_pixel_acc, classwise_gtpixels, classwise_predpixels = prediction_stat([x], labels, self.n_classes)

            # Need to perform this operation for MultiGPU
            classwise_pixel_acc = Variable(torch.FloatTensor([classwise_pixel_acc]).cuda())
            classwise_gtpixels = Variable(torch.FloatTensor([classwise_gtpixels]).cuda())
            classwise_predpixels = Variable(torch.FloatTensor([classwise_predpixels]).cuda())

            return x, classwise_pixel_acc, classwise_gtpixels, classwise_predpixels
        else:
            return x
        '''
