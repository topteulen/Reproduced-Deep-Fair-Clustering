import torchvision
from torchvision import transforms
import torch
import torch.utils.data as data
import numpy as np
import os

root='./data/digital/'
batch_size=128
num_workers=0
#download from https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass.html#usps
download=False
dataset = torchvision.datasets.USPS(
    root, train=True, download=download)


os.mkdir(root + '/train_usps/')
for idx, (img, _) in enumerate(dataset):
    img.save( root + '/train_usps/'+'{:05d}.jpg'.format(idx))
    