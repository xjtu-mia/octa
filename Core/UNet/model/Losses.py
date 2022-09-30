import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np


class SoftDiceLoss(nn.Module):
    def __init__(self, classes):
        super(SoftDiceLoss, self).__init__()
        self.classes = classes

    def forward(self, input, target):
        smooth = 0.01
        batch_size = input.size(0)
        input = torch.softmax(input, dim=1).view(batch_size, self.classes, -1)
        target = (255*target).long()   #long()
        target[target == 127] = 1
        target[target == 255] = 2
        #target = self.one_hot_encoder(target).contiguous().view(batch_size, self.classes, -1)
        target = nn.functional.one_hot(target, self.classes).permute(0, -1, *range(1, target.dim())).squeeze(dim=2).float()
        target = target.contiguous().view(batch_size, self.classes, -1)
        inter = torch.sum(input * target, 2) + smooth
        union = torch.sum(input, 2) + torch.sum(target, 2) + smooth

        score = torch.sum(2.0 * inter / union)
        score = 1.0 - score / (float(batch_size) * float(self.classes))

        return score


def calc_loss(prediction, target, n_classes=3):
    dicemetric = SoftDiceLoss(classes=n_classes)

    loss = dicemetric(prediction, target)
    target = (255 * target).long()  # long()
    target[target == 127] = 1
    target[target == 255] = 2
    input = prediction.transpose(1, 2).transpose(2, 3).contiguous().view(-1, 3)
    target = target.contiguous().view(-1)
    ce_loss = nn.CrossEntropyLoss(weight=None, size_average=True, reduce=False)
    ce_loss = ce_loss(input, target)
    ce_loss = torch.mean(ce_loss)
    # print(loss,'---ce:', ce_loss)
    return 0.7*loss + 0.3*ce_loss
    # return ce_loss


class Dice(nn.Module):
    def __init__(self, classes):
        super(Dice, self).__init__()
        self.classes = classes

    def forward(self, input, target):

        batch_size = input.size(0)
        input = torch.softmax(input, dim=1)
        # input = nn.functional.one_hot(torch.argmax(input, dim=1).unsqueeze(dim=1), self.classes)
        # print(input.size())
        input = nn.functional.one_hot(torch.argmax(input, dim=1).unsqueeze(dim=1), self.classes).permute(0, -1, *range(1, target.dim())).squeeze(dim=2).float()

        input = input.contiguous().view(batch_size, self.classes, -1)
        target = (255*target).long()   #long()
        target[target == 127] = 1
        target[target == 255] = 2
        #target = self.one_hot_encoder(target).contiguous().view(batch_size, self.classes, -1)
        target = nn.functional.one_hot(target, self.classes).permute(0, -1, *range(1, target.dim())).squeeze(dim=2).float()

        # print(target.size())
        target = target.contiguous().view(batch_size, self.classes, -1)
        inter = torch.sum(input * target, 2)
        union = torch.sum(input, 2) + torch.sum(target, 2)

        score = torch.sum(2.0 * inter / union)
        score = score / (float(batch_size) * float(self.classes))

        return score


def calc_dice(prediction, target, n_classes=3):
    dicemetric = Dice(classes=n_classes)

    dice = dicemetric(prediction, target)
    return dice