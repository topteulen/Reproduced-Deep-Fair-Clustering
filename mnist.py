import torchvision
from torchvision import transforms
import torch
import torch.utils.data as data
import numpy as np
import os

root='./data/digital/'
batch_size=128
num_workers=0
download=True
dataset = torchvision.datasets.MNIST(
    root, train=True, download=download)


os.mkdir(root + '/train_mnist/')
for idx, (img, _) in enumerate(dataset):
    img.save( root + '/train_mnist/'+'{:05d}.jpg'.format(idx))
    