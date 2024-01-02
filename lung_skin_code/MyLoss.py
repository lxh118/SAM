# _*_ coding : utf-8 _*_
# @Time : 2023/12/26 13:24
# @Author : 娄星华
# @File : MyLoss
# @Project : SAM

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, Inputs, Targets):
        BCE_loss = F.binary_cross_entropy_with_logits(Inputs, Targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return torch.mean(focal_loss)


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, Inputs, Targets):
        intersection = torch.sum(Inputs * Targets)
        dice_coefficient = (2.0 * intersection + self.smooth) / (torch.sum(Inputs) + torch.sum(Targets) + self.smooth)
        dice_loss = 1 - dice_coefficient
        return dice_loss


class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5):
        super(CombinedLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.focal_loss = FocalLoss()
        self.dice_loss = DiceLoss()

    def forward(self, Inputs, Targets):
        return self.alpha * self.focal_loss(Inputs, Targets) + self.beta * self.dice_loss(Inputs, Targets)


if __name__ == "__main__":
    # Example usage:
    # Assuming inputs and targets are your predicted and ground truth tensors
    inputs = torch.tensor([[0.2, 0.8, 0.3], [0.7, 0.9, 0.4], [0.1, 0.5, 0.6]], requires_grad=True)
    targets = torch.tensor([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=torch.float32)

    # Instantiate the combined loss
    combined_loss = CombinedLoss(alpha=0.5, beta=0.5)

    # Calculate the loss
    loss = combined_loss(inputs, targets)
    print(loss.item())
    pass
