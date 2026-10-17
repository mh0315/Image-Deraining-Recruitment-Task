import torch
import torch.nn as nn
from torchvision.models import vgg19, VGG19_Weights

class easy_contrastive_loss(nn.Module):
    def __init__(self):
        super(easy_contrastive_loss,self).__init__()
        vgg = vgg19(weights=VGG19_Weights.DEFAULT).features
        for param in vgg.parameters():
            param.requires_gard = False
        self.layers = nn.Sequential(*(list(vgg)[:35])).eval()

        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.cos = nn.CosineSimilarity(dim=1)

    def forward(self, generated, ground_truth,rainy):
        feat1 = self.pool(self.layers(generated)).flatten(1)
        feat2 = self.pool(self.layers(ground_truth)).flatten(1)
        feat3 = self.pool(self.layers(rainy)).flatten(1)
                
        sim1 = self.cos(feat1,feat2)
        sim2 = self.cos(feat1,feat3)

        Loss = (1 - sim1) + sim2
        return Loss.mean()