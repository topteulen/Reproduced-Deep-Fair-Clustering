import torchvision
from torchvision import transforms
import torch
import torch.utils.data as data
import numpy as np
import os

root='./data/digital'
batch_size=128
num_workers=0
download=True
dataset = torchvision.datasets.MNIST(
    root, train=True, download=download)
    
if not os.path.exists(root + '/train_mnist/'):
    os.mkdir(root + '/train_mnist/')
f = open("./data/train_mnist.txt", "a",newline='\n')
for idx, (img, label) in enumerate(dataset):
    f.write(root + '/train_mnist/'+'{:05d}.jpg'.format(idx) + " " + str(label) + "\n" )
    img.save( root + '/train_mnist/'+'{:05d}.jpg'.format(idx))
    
