import torchvision
from torchvision import transforms
from PIL import Image
import torch
import torch.utils.data as data
import numpy as np
import os

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
dataset = torchvision.datasets.MNIST(
    root, train=True,transform=transform,download=download)
    
if not os.path.exists(root + '/train_Rmnist/'):
    os.mkdir(root + '/train_Rmnist/')
f = open("./data/train_Rmnist.txt", "a",newline='\n')
for idx, (img, label) in enumerate(dataset):
    f.write(root + '/train_Rmnist/'+'{:05d}.jpg'.format(idx) + " " + str(label) + "\n" )
    img.save( root + '/train_Rmnist/'+'{:05d}.jpg'.format(idx))
    
