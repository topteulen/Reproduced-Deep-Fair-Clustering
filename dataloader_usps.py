"""
This module downloads and loads the mnist dataset in to the correct folder
"""
import torchvision
from torchvision import transforms
import torch
import torch.utils.data as data
import numpy as np
import os

#Set path where you want data to be stored
root='./data/digital'
batch_size=128
num_workers=0
download=True
#load data from torchvision
dataset = torchvision.datasets.USPS(
    "./data/", train=True, download=download)

#make dir   
if not os.path.exists(root + '/train_usps/'):
    os.mkdir(root + '/train_usps/')
f = open("./data/train_usps.txt", "a",newline='\n')
#create files
for idx, (img, label) in enumerate(dataset):
    f.write(root + '/train_usps/'+'{:05d}.jpg'.format(idx) + " " + str(label)+"\n")
    img.save( root + '/train_usps/'+'{:05d}.jpg'.format(idx))
    
# with open("./data/train_usps.txt", 'r') as file:
#     content = file.read()
# 
# with open("./data/train_usps.txt", 'w', newline='\n') as file:
#     file.write(content)