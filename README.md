# MIAdam

This repository contains the source code of MIAdam optimizer from our paper, A Method for Enhancing Generalization of Adam by Multiple Integrations.

## Image classification experiments on CIFAR using MIAdam

```shell
python train/train_cifar10.py --arch r18 --opt  miadam1 --lr 1e-3 -bs 128 --epoch 150 -kappa 0.98  --sub_epoch 20  
python train/train_cifar10.py --arch r18 --opt  miadam2 --lr 1e-3 -bs 128 --epoch 150 -kappa 0.98  --sub_epoch 20  
python train/train_cifar10.py --arch r18 --opt  miadam3 --lr 1e-3 -bs 128 --epoch 150 -kappa 0.98  --sub_epoch 20  

```

## Experiments on Datasets Injected with Label Noises

```shell
python train/train_cifar10_label_noise.py --arch r18 --opt  miadam1 --lr 1e-3 -bs 128 --epoch 150 -kappa 0.98  --sub_epoch 20  -noise 0.2
```

## Environment

- NVIDIA GeForce RTX 2080Ti GPU
- Python 3.6.0
- Pytorch 1.8.3
