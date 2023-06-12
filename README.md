# 介绍

这是本科毕设模型 Cycle-Consistent Generative Adversarial Networks with  High Definition（CycleGANHD）的代码实现

该模型实现了实时的将图像风格化



# 使用

默认情况下，输入文件在`default_models/images/inputs`文件夹，输出文件在`default_models/images/outputs`文件夹

运行`main.py`文件，将输入文件夹下的图片风格化输出到输出文件夹

```python
python main.py 
```

具体可选参数请查看help

```python
python main.py -h
```



# 训练集下载

请从 https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/ 下载需要训练的风格数据集解压至`datasets`文件夹



# 训练自己的权重

训练运行`train.py`文件

```python
python train.py
```

具体可选参数请查看help

```python
python train.py -h
```

