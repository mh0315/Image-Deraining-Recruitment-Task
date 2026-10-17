import torch
import torch.nn as nn
from torchvision.models import vgg19, VGG19_Weights


class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        vgg = vgg19(weights=VGG19_Weights.DEFAULT).features
        for param in vgg.parameters():
            param.requires_grad = False
        self.layers = nn.Sequential(*(list(vgg)[:35])).eval()

        self.loss = nn.L1Loss()
    def forward(self, generated, ground_truth):
        feature1 = self.layers(generated)
        feature2 = self.layers(ground_truth)

        Loss = self.loss(feature1,feature2) 
        return Loss