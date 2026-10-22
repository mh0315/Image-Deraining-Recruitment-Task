# Part2 - 第二部分 - 任务一

## 基于像素到底有什么不好

1.当图像过于平滑时， PSNR、SSIM得出的结果会和人眼感知得出的结果产生较大的差距，这是因为这两者本质上只是关注像素在数值上的差距。而 **LPIPS**不是关注图片像素值本身的差距，它更关注特征图之间的差距，此时的L1计算的是由预训练模型获得的**特征图中像素之间的距离**，**特征图上的每一个像素，其实代表着原图上的一个“感受野”**。

2.预训练模型在这里**用来提取原图和生成图的深度特征，把像素值转变成与人眼感知相似的特征，用来计算LPIPS**。

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

