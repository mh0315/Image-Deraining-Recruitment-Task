# Part2 - 第一部分 任务二

1.下面为我的模型代码

```python
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
    
```

2.下图为训练结果![](../Image-Deraining-Recruitment-Task/Reports/Part2 第一部分 任务二 3 .png)

**3.思考**

 **1**.DerainNet先进行下采样是为了**提取特征**，然后再进行上采样则是为了**恢复到原来的尺寸与原图能进行计算**。

**下采样**就是把图片的尺寸减少,可以通过池化操作来进行；**上采样**就是把图片的尺寸增加，常用插值法和转置卷积。下面是一个使用插值法的例子：

```python
up = F.interpolate(out, size=(H, W), mode='bilinear', align_corners=False)
```

`interpolate`是pytorch中专门用来缩放特征图尺寸的插值函数，`size=()`中输入H和W

`mode='bilinear'`是双线性插值，根据周围4个像素加权平均算出新像素

`align_corners=False`可以避免边缘像素扭曲

**concat**指的是将多个张量拼接在一起成为一个大张量，必须满足**除拼接维度外，其他所有维度必须完全一致**，以下是一个示例：

```python
concat = torch.cat([up_1, up_2, up_3], dim=1)
```

`dim`表示沿哪个维度，[0,1,2,3]分别对应[B,C,H,W]

 **2.学到的知识**

当遇到层数较多的神经网络时，不用每一层都写出来，可以用下面之中方法：

```python
class SimpleDynamicNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=16, num_layers=3):
        super().__init__()
        
        layers = []
        
        for i in range(num_layers):
           
            current_in = in_channels if i == 0 else out_channels
            
            layers.append(nn.Conv2d(current_in, out_channels, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
        
        self.body = nn.Sequential(*layers)
        
        self.out_conv = nn.Conv2d(out_channels, 3, kernel_size=3, padding=1)

```



`layers = []`准备一个空列表，存放动态生成的层

`for i in range(n)`循环n次

`current_in = in_channels if i == 0 else out_channels`

第一层接受in_channels通道的图片，其余层接受out_channels通道的层

`layers.append(nn.Conv2d(current_in, out_channels, kernel_size=3, padding=1))`

`layers.append(nn.BatchNorm2d(out_channels)`

`layers.append(nn.ReLU()`

append的意思是追加到列表末尾

`self.body = nn.Sequential(*layers)`*layers是python的解包写法，把列表中的层拆出来

`nn.Sequential(...)`接受拆出来的层，按顺序打包







