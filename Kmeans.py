import random
import numpy as np
import torch
from torch import nn
import sklearn.cluster as sk
from dataloader import mnist_usps
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--bs", type=int, default=512)
parser.add_argument("--k", type=int, default=10)
parser.add_argument("--lr", type=float, default=1e-2)
parser.add_argument("--iters", type=int, default=20000)
parser.add_argument("--test_interval", type=int, default=5000)
parser.add_argument("--adv_mult", type=float, default=10.0)
parser.add_argument("--coeff_fair", type=float, default=1.0)
parser.add_argument("--coeff_par", type=float, default=1.0)
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--seed", type=int, default=2019)
parser.add_argument("--corrupted",type=float, default=0)
args = parser.parse_args()

POOL = nn.MaxPool2d(4)
args.bs = 512
data_loader = mnist_usps(args)
kmeans = sk.MiniBatchKMeans(10,max_iter=1000,batch_size=args.bs)
len_image_0 = len(data_loader[0])
len_image_1 = len(data_loader[1])
for step in range(int(60000/args.bs)):
    if step % len_image_0 == 0:
        iter_image_0 = iter(data_loader[0])
    if step % len_image_1 == 0:
        iter_image_1 = iter(data_loader[1])
        
    image_0, a = iter_image_0.__next__()
    image_1, _ = iter_image_1.__next__()
    image_0 = POOL(image_0)
    image_1 = POOL(image_1)
    if step == 0 :
        first_images = image_0
        first_labels = a
    
    image_0 = image_0.reshape(args.bs,-1)
    image_1 = image_1.reshape(args.bs,-1)
    kmeans = kmeans.partial_fit(image_0)
    kmeans = kmeans.partial_fit(image_1)
    if step % 10 == 0:
        print(step, int(60000/args.bs))
clusters = kmeans.cluster_centers_
open("kmeans.txt", "w")
file = open("kmeans.txt", "a")
for i in range(len(clusters)):
    file.writelines([str(item)+" " for item in clusters[i]])
    file.writelines("\n")
    



first_images = first_images.reshape(args.bs,-1)
print(kmeans.predict(first_images.numpy()))
print(first_labels)
correct = first_labels.numpy() == kmeans.predict(first_images.numpy())
print("acc",sum(correct)/len(first_labels))