import torch
import torch.nn as nn
import torch.nn.functional as F

class Resblock(nn.Module):
    def __init__(self,in_channels,out_channels,stride=1):
        super().__init__()
        layers = []
        for i in range(8):
            current_in = in_channels if i == 0 else out_channels
            layers.append(nn.Conv2d(current_in, out_channels, kernel_size=3, stride=1, padding=1))
            layers.append(nn.BatchNorm2d(out_channels))
            if i<7:
                layers.append(nn.ReLU())
            self.body = nn.Sequential(*layers)
            self.relu = nn.ReLU()

    def forward(self,x):
        x = self.body(x)+x
        x = self.relu(x)
        return x
    

class DerainNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Conv2d(3,64,3,1,1)
        self.resblock1 = Resblock(64,64)
        self.resblock2 = Resblock(64,64)
        self.resblock3 = Resblock(64,64)

        self.fusion = nn.Sequential(
            nn.Conv2d(192,64,3,1,1),
            nn.ReLU(),
            nn.Conv2d(64,3,3,1,1)

        )

    def forward(self,x):
        H,W = x.shape[2],x.shape[3]
        x0 = self.head(x)
        down_x1 = x0
        down_x2 = F.max_pool2d(x0, kernel_size=2,stride=2)
        down_x3 = F.max_pool2d(x0, kernel_size=4,stride=4)

        out1 = self.resblock1(down_x1)
        out2 = self.resblock2(down_x2)
        out3 = self.resblock3(down_x3)

        up_1 = F.interpolate(out1,size=(H,W),mode='bilinear',align_corners=False)
        up_2 = F.interpolate(out2,size=(H,W),mode='bilinear',align_corners=False)
        up_3 = F.interpolate(out3,size=(H,W),mode='bilinear',align_corners=False)

        concat = torch.cat([up_1, up_2, up_3], dim=1)
        rain = self.fusion(concat)
        clean = x - rain
        return clean
    