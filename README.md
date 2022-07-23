# MicroNet: Improving Image Recognition with Extremely Low FLOPs (ICCV 2021)
[MicroNet](https://arxiv.org/abs/2108.05894)的一个 [pytorch](http://pytorch.org/) 的实现

# 环境要求

- Linux or macOS with Python ≥ 3.6.
- *Anaconda3*, *PyTorch ≥ 1.5* with matched [torchvision](https://github.com/pytorch/vision/)

经过测试的可用环境：
- ubutnu
- ptyhon3.9
- pytorh1.11


# 不同型号模型的表现及预训练模型下载
Model | #Param | MAdds | Top-1 | download
--- |:---:|:---:|:---:|:---:
MicroNet-M3 | 2.6M | 21M  | 62.5 | [model](http://www.svcl.ucsd.edu/projects/micronet/assets/micronet-m3.pth)
MicroNet-M2 | 2.4M | 12M  | 59.4 | [model](http://www.svcl.ucsd.edu/projects/micronet/assets/micronet-m2.pth)
MicroNet-M1 | 1.8M | 6M  | 51.4 | [model](http://www.svcl.ucsd.edu/projects/micronet/assets/micronet-m1.pth)
MicroNet-M0 | 1.0M | 4M  | 46.6 | [model](http://www.svcl.ucsd.edu/projects/micronet/assets/micronet-m0.pth)

# 在ImageNet上验证MicroNet
首先下载预训练模型，然后使用相应的sh脚本启动验证程序。比如你想验证MicroNet-M3，可以使用一下命令：
```
sh scripts/eval_micronet_m3.sh /path/to/imagenet /path/to/output /path/to/pretrained_model
```

注意：比如你的imagenet val如下所示，那么/path/to/imagenet=....../n1k/
![image](https://user-images.githubusercontent.com/48787805/180604378-ea2b1d39-3dda-43ba-82d4-0a978df91e5c.png)




