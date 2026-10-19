# Part2 - 第二部分 - 任务一

## 基于像素到底有什么不好

1.因为LPIPS在训练过程中模型寻找的是两张图之间的**特征关系**，比较的**对象变成了特征**，而PSNR、SSIM比较的**对象都是像素**。

2.预训练模型在这里用来**提取图像特征，为如何处理图像提供标准**。

3.

 1.以下为补全的代码

```python
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
```



 2.训练结果![](<Part2 第二部分 任务一 train.png>)

测试结果

![](<Part2 第二部分 任务一 test.png>)

