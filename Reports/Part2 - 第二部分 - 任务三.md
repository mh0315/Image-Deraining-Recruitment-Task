# Part2 - 第二部分 - 任务三

## 改造感知损失函数，使用特征做对比

**2.在 `easy contrastive loss` 文件中编写新的损失函数：**

 AdaptiveAvgPool2d用法例：

```python
pool = nn.AdaptiveAvgPool2d((1,1))
```

`(1,1)`表示将H，W压缩成1,1

```python
feat_vector = feat_pooled.flatten(1)
```

去除多余的维度，只留下（B，C）

`flatten(1)`表示从第一维开始展平

可以使用` nn.CosineSimilarity(dim=1)`来计算余弦相似度

**3.使用新的损失函数训练并测试模型**

训练结果

![](../Image-Deraining-Recruitment-Task/Reports/Part2 第二部分 任务三 train.png)

测试结果

![](../Image-Deraining-Recruitment-Task/Reports/Part2 第二部分 任务三 test.png)

