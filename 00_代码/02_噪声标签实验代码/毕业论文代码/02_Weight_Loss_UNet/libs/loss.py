from __future__ import print_function, division
from torch.nn.modules.loss import  _Loss
import torch.nn.functional as F
import torch

class WeightedCELoss(_Loss):
    def __init__(self, size_average=True, reduce=True):
        super(WeightedCELoss, self).__init__(size_average, reduce)

    def forward(self, input, target,weight):   
        n, c, h, w = input.size()
        log_p = F.log_softmax(input, dim=1)
        return F.nll_loss(log_p, target.long(),weight=torch.mean(weight,dim=0), reduce=False)

