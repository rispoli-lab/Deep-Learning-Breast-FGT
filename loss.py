import torch
import torch.nn as nn

def FGT_coe(FGT,volume):
    FGT_p = FGT.sum()/volume.sum()
    return FGT_p

def dice_coef(pred, target):
    #pred = pred.contiguous()
    #target = target.contiguous()

    intersection = (pred * target).sum()

    dice_coe = ((2. * intersection ) / (pred.sum() + target.sum() ))


    return dice_coe

def dice_loss(pred, target, smooth=1.):
    pred = pred.contiguous()
    target = target.contiguous()

    intersection = (pred * target).sum(dim=1).sum(dim=1)

    loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=1).sum(dim=1) + target.sum(dim=1).sum(dim=1) + smooth)))


    return loss.mean()