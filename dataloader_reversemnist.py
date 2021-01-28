"""
This module transforms or downloads and transforms the Mnist dataset
By inverting the colours to make the reverse mnist dataset.
"""

import torchvision
from torchvision import transforms
from PIL import Image
import torch
import torch.utils.data as data
import numpy as np
import os

#Set path where you want data to be stored
root='./data/digital'
batch_size=128
num_workers=0
download=True

class InverseTransform:
    """Inverse images."""
    def __init__(self):
        pass
    def __call__(self, img):
        img = np.array(img)
        img = 255-img
        img = Image.fromarray(img)
        return img
        
transform = transforms.Compose([InverseTransform(),    
                            # transforms.Normalize((0.5,), (0.5,))
                            ])
#load data from torchvision
dataset = torchvision.datasets.MNIST(
    root, train=True,transform=transform,download=download)
#make dir   
if not os.path.exists(root + '/train_Rmnist/'):
    os.mkdir(root + '/train_Rmnist/')
f = open("./data/train_Rmnist.txt", "a",newline='\n')
#create files
for idx, (img, label) in enumerate(dataset):
    f.write(root + '/train_Rmnist/'+'{:05d}.jpg'.format(idx) + " " + str(label) + "\n" )
    img.save( root + '/train_Rmnist/'+'{:05d}.jpg'.format(idx))
    
