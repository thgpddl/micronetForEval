
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


# 亲测验证
环境配置见requirement.txt
使用的是ubuntu+2080ti环境
目录结构如下：

![image](https://user-images.githubusercontent.com/48787805/180638461-5d2a61aa-345d-4a20-b2d2-2fb9ceb44f3c.png)

执行命令:
`sh scripts/eval_micronet_m3.sh dataset/n1K output micronet-m3.pth `开始验证，结果如下：

```bash
$ sh scripts/eval_micronet_m3.sh dataset/n1K output micronet-m3.pth 
Please install DALI from https://www.github.com/NVIDIA/DALI to run this example.
=> creating model 'MicroNet'
[0, 2, 0]
reduction: 8, squeeze: 48/8
init-a: [1.0, 0.5], init-b: [0.0, 0.5]
group shuffle: 12, divide group: False
inp: 48, oup:16, g:4
[0, 2, 0]
reduction: 8, squeeze: 64/8
init-a: [1.0, 0.5], init-b: [0.0, 0.5]
group shuffle: 16, divide group: False
inp: 64, oup:24, g:4
[0, 2, 0]
reduction: 8, squeeze: 96/12
init-a: [1.0, 0.5], init-b: [0.0, 0.5]
group shuffle: 24, divide group: False
inp: 96, oup:24, g:4
[0, 2, 0]
inp: 24, oup:144, g:6
reduction: 8, squeeze: 144/20
init-a: [1.0, 0.5], init-b: [0.0, 0.5]
group shuffle: 24, divide group: True
inp: 144, oup:32, g:4
[0, 2, 0]
inp: 32, oup:192, g:8
reduction: 16, squeeze: 192/12
init-a: [1.0, 0.5], init-b: [0.0, 0.5]
group shuffle: 24, divide group: True
inp: 192, oup:32, g:4
[0, 2, 0]
inp: 32, oup:192, g:8
reduction: 16, squeeze: 192/12
init-a: [1.0, 0.5], init-b: [0.0, 0.5]
group shuffle: 24, divide group: True
inp: 192, oup:64, g:8
[0, 2, 0]
inp: 64, oup:384, g:8
reduction: 16, squeeze: 384/24
init-a: [1.0, 0.5], init-b: [0.0, 0.5]
group shuffle: 48, divide group: True
inp: 384, oup:80, g:8
[0, 2, 0]
inp: 80, oup:480, g:10
reduction: 16, squeeze: 480/32
init-a: [1.0, 0.5], init-b: [0.0, 0.5]
group shuffle: 48, divide group: True
inp: 480, oup:80, g:8
[0, 2, 0]
inp: 80, oup:480, g:10
reduction: 16, squeeze: 480/32
init-a: [1.0, 0.5], init-b: [0.0, 0.5]
group shuffle: 48, divide group: True
inp: 480, oup:120, g:10
[0, 2, 0]
inp: 120, oup:720, g:12
reduction: 16, squeeze: 720/44
init-a: [1.0, 0.5], init-b: [0.0, 0.5]
group shuffle: 60, divide group: True
inp: 720, oup:120, g:10
[0, 2, 0]
inp: 120, oup:720, g:12
reduction: 16, squeeze: 720/44
init-a: [1.0, 0.5], init-b: [0.0, 0.5]
group shuffle: 60, divide group: True
inp: 720, oup:144, g:12
[0, 2, 0]
inp: 144, oup:864, g:12
output/micronet-m3-eval/arch.txt
/usr/local/lib/python3.9/dist-packages/torch/utils/data/dataloader.py:487: UserWarning: This DataLoader will create 48 worker processes in total. Our suggested max number of worker in current system is 24, which is smaller than what this DataLoader is going to create. Please be aware that excessive worker creation might get DataLoader running slow or even freeze, lower the worker number to avoid potential slowness/freeze if necessary.
  warnings.warn(_create_warning_msg(
=> loading pretrained weight 'micronet-m3.pth'
correct.shape: torch.Size([5, 512])
correct.shape: torch.Size([5, 512])
Test: [0/98]    Time 25.769 (25.769)    Loss 2.3842 (2.3842)    Acc@1 76.172 (76.172)   Acc@5 92.578 (92.578)
correct.shape: torch.Size([5, 512])
correct.shape: torch.Size([5, 512])
correct.shape: torch.Size([5, 512])
correct.shape: torch.Size([5, 512])
correct.shape: torch.Size([5, 512])
correct.shape: torch.Size([5, 512])
correct.shape: torch.Size([5, 512])
correct.shape: torch.Size([5, 512])
correct.shape: torch.Size([5, 512])
correct.shape: torch.Size([5, 512])
correct.shape: torch.Size([5, 512])
correct.shape: torch.Size([5, 512])
correct.shape: torch.Size([5, 512])
correct.shape: torch.Size([5, 512])
correct.shape: torch.Size([5, 512])
correct.shape: torch.Size([5, 512])
correct.shape: torch.Size([5, 512])
correct.shape: torch.Size([5, 512])
correct.shape: torch.Size([5, 512])
correct.shape: torch.Size([5, 512])
correct.shape: torch.Size([5, 512])
correct.shape: torch.Size([5, 512])
correct.shape: torch.Size([5, 512])
correct.shape: torch.Size([5, 512])
correct.shape: torch.Size([5, 512])
correct.shape: torch.Size([5, 512])
correct.shape: torch.Size([5, 512])
correct.shape: torch.Size([5, 512])
correct.shape: torch.Size([5, 512])
correct.shape: torch.Size([5, 512])
correct.shape: torch.Size([5, 512])
correct.shape: torch.Size([5, 512])
correct.shape: torch.Size([5, 512])
correct.shape: torch.Size([5, 512])
correct.shape: torch.Size([5, 512])
correct.shape: torch.Size([5, 512])
correct.shape: torch.Size([5, 512])
correct.shape: torch.Size([5, 512])
correct.shape: torch.Size([5, 512])
correct.shape: torch.Size([5, 512])
correct.shape: torch.Size([5, 512])
correct.shape: torch.Size([5, 512])
correct.shape: torch.Size([5, 512])
correct.shape: torch.Size([5, 512])
correct.shape: torch.Size([5, 512])
correct.shape: torch.Size([5, 512])
correct.shape: torch.Size([5, 512])
correct.shape: torch.Size([5, 512])
correct.shape: torch.Size([5, 512])
correct.shape: torch.Size([5, 512])
correct.shape: torch.Size([5, 512])
correct.shape: torch.Size([5, 512])
correct.shape: torch.Size([5, 512])
correct.shape: torch.Size([5, 512])
correct.shape: torch.Size([5, 512])
correct.shape: torch.Size([5, 512])
correct.shape: torch.Size([5, 512])
correct.shape: torch.Size([5, 512])
correct.shape: torch.Size([5, 512])
correct.shape: torch.Size([5, 512])
correct.shape: torch.Size([5, 512])
correct.shape: torch.Size([5, 512])
correct.shape: torch.Size([5, 512])
correct.shape: torch.Size([5, 512])
correct.shape: torch.Size([5, 512])
correct.shape: torch.Size([5, 512])
correct.shape: torch.Size([5, 512])
correct.shape: torch.Size([5, 512])
correct.shape: torch.Size([5, 512])
correct.shape: torch.Size([5, 512])
correct.shape: torch.Size([5, 512])
correct.shape: torch.Size([5, 512])
correct.shape: torch.Size([5, 512])
correct.shape: torch.Size([5, 512])
correct.shape: torch.Size([5, 512])
correct.shape: torch.Size([5, 512])
correct.shape: torch.Size([5, 512])
correct.shape: torch.Size([5, 512])
correct.shape: torch.Size([5, 512])
correct.shape: torch.Size([5, 512])
correct.shape: torch.Size([5, 512])
correct.shape: torch.Size([5, 512])
correct.shape: torch.Size([5, 512])
correct.shape: torch.Size([5, 512])
correct.shape: torch.Size([5, 512])
correct.shape: torch.Size([5, 512])
correct.shape: torch.Size([5, 512])
correct.shape: torch.Size([5, 512])
correct.shape: torch.Size([5, 512])
correct.shape: torch.Size([5, 512])
correct.shape: torch.Size([5, 512])
correct.shape: torch.Size([5, 512])
correct.shape: torch.Size([5, 512])
correct.shape: torch.Size([5, 512])
correct.shape: torch.Size([5, 512])
correct.shape: torch.Size([5, 512])
correct.shape: torch.Size([5, 512])
correct.shape: torch.Size([5, 512])
correct.shape: torch.Size([5, 512])
correct.shape: torch.Size([5, 512])
correct.shape: torch.Size([5, 512])
correct.shape: torch.Size([5, 512])
correct.shape: torch.Size([5, 512])
correct.shape: torch.Size([5, 512])
correct.shape: torch.Size([5, 512])
correct.shape: torch.Size([5, 512])
correct.shape: torch.Size([5, 512])
correct.shape: torch.Size([5, 512])
correct.shape: torch.Size([5, 512])
correct.shape: torch.Size([5, 512])
correct.shape: torch.Size([5, 512])
correct.shape: torch.Size([5, 512])
correct.shape: torch.Size([5, 512])
correct.shape: torch.Size([5, 512])
correct.shape: torch.Size([5, 512])
correct.shape: torch.Size([5, 512])
correct.shape: torch.Size([5, 512])
correct.shape: torch.Size([5, 512])
correct.shape: torch.Size([5, 512])
correct.shape: torch.Size([5, 512])
correct.shape: torch.Size([5, 512])
correct.shape: torch.Size([5, 512])
correct.shape: torch.Size([5, 512])
correct.shape: torch.Size([5, 512])
correct.shape: torch.Size([5, 512])
correct.shape: torch.Size([5, 512])
correct.shape: torch.Size([5, 512])
correct.shape: torch.Size([5, 512])
correct.shape: torch.Size([5, 512])
correct.shape: torch.Size([5, 512])
correct.shape: torch.Size([5, 512])
correct.shape: torch.Size([5, 512])
correct.shape: torch.Size([5, 512])
correct.shape: torch.Size([5, 512])
correct.shape: torch.Size([5, 512])
correct.shape: torch.Size([5, 512])
correct.shape: torch.Size([5, 512])
correct.shape: torch.Size([5, 512])
correct.shape: torch.Size([5, 512])
correct.shape: torch.Size([5, 512])
correct.shape: torch.Size([5, 512])
correct.shape: torch.Size([5, 512])
correct.shape: torch.Size([5, 512])
correct.shape: torch.Size([5, 512])
correct.shape: torch.Size([5, 512])
correct.shape: torch.Size([5, 512])
correct.shape: torch.Size([5, 512])
correct.shape: torch.Size([5, 512])
correct.shape: torch.Size([5, 512])
correct.shape: torch.Size([5, 512])
correct.shape: torch.Size([5, 512])
correct.shape: torch.Size([5, 512])
correct.shape: torch.Size([5, 512])
correct.shape: torch.Size([5, 512])
correct.shape: torch.Size([5, 512])
correct.shape: torch.Size([5, 512])
correct.shape: torch.Size([5, 512])
correct.shape: torch.Size([5, 512])
correct.shape: torch.Size([5, 512])
correct.shape: torch.Size([5, 512])
correct.shape: torch.Size([5, 512])
correct.shape: torch.Size([5, 512])
correct.shape: torch.Size([5, 512])
correct.shape: torch.Size([5, 512])
correct.shape: torch.Size([5, 512])
correct.shape: torch.Size([5, 512])
correct.shape: torch.Size([5, 512])
correct.shape: torch.Size([5, 512])
correct.shape: torch.Size([5, 512])
correct.shape: torch.Size([5, 512])
correct.shape: torch.Size([5, 512])
correct.shape: torch.Size([5, 512])
correct.shape: torch.Size([5, 512])
correct.shape: torch.Size([5, 512])
correct.shape: torch.Size([5, 512])
correct.shape: torch.Size([5, 512])
correct.shape: torch.Size([5, 512])
correct.shape: torch.Size([5, 512])
correct.shape: torch.Size([5, 512])
correct.shape: torch.Size([5, 512])
correct.shape: torch.Size([5, 512])
correct.shape: torch.Size([5, 512])
correct.shape: torch.Size([5, 512])
correct.shape: torch.Size([5, 512])
correct.shape: torch.Size([5, 512])
correct.shape: torch.Size([5, 512])
correct.shape: torch.Size([5, 512])
correct.shape: torch.Size([5, 512])
correct.shape: torch.Size([5, 512])
correct.shape: torch.Size([5, 512])
correct.shape: torch.Size([5, 512])
correct.shape: torch.Size([5, 512])
correct.shape: torch.Size([5, 336])
correct.shape: torch.Size([5, 336])
 * Acc@1 62.496 Acc@5 83.106
```




