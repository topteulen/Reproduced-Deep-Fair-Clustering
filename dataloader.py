# @Author  : Peizhao Li
# @Contact : peizhaoli05@gmail.com
"""
This module loads the datasets into memory and applies resizing and preprocessing.
Also used to aply corruption if that extentsion is activated.
"""
from PIL import Image
import numpy as np

import torchvision.transforms.functional as TF
import random
import torch
from torch.utils import data
from torchvision import transforms

kwargs = {"shuffle": True, "num_workers": 0, "pin_memory": True, "drop_last": True}


class digital(data.Dataset):
    #load images from textfile with size in 32x32
    def __init__(self, subset, transform=None):
        file_dir = "./data/{}.txt".format(subset)
        self.data_dir = open(file_dir).readlines()
        self.transform = transform
    #get path and retrieve image object
    def __getitem__(self, index):
        img_dir, label = self.data_dir[index].split()
        img = Image.open(img_dir)
        if self.transform is not None:
            img = self.transform(img)
        label = torch.tensor(np.int64(label)).long()

        return img, label

    def __len__(self):
        return len(self.data_dir)



class CorruptionTransform:
    """Corrupt by certain precentage."""
    #set percentages
    def __init__(self, percent):
        self.percent = percent
    #corrupt the images and return img object
    def __call__(self, img):
        #cast to array
        img = np.array(img)
        base_color = img[0][0]
        length = len(img)
        #corrupt given certain chance to oposite colour
        for x in range(length):
            chance = np.random.rand(length)
            img[x][chance < self.percent] = 255-base_color
        #make PIL object again
        img = Image.fromarray(img)
        return img




def get_digital(args, subset,resize=[0],colour=0,corruption=False):
    """retrieves images from drive and converts them to PIL objects.
    can corrupt if flag is given"""
    if corruption == True:
        #checks if valid percentage of corruption
        if 1 > args.corrupted > 0:
            transform = transforms.Compose([transforms.Pad(resize[0],colour),
                                            CorruptionTransform(args.corrupted),
                                            transforms.ToTensor(),
                                        ])
        else:
            print("give args.corrupted a correct values please")
    else:
        transform = transforms.Compose([transforms.Pad(resize[0],colour),
                                        transforms.ToTensor(),
                                        ])

    #preforms the collection and transformation defined above
    data = digital(subset, transform)
    data_loader = torch.utils.data.DataLoader(
        dataset=data,
        batch_size=args.bs,
        **kwargs
    )

    return data_loader


def mnist_usps(args):
    """loading in the mnist usps dataset with the correct dataset corruption if 
    specified."""
    if args.corrupted != 0:
        if  args.corrupted_set == 0:
            train_0 = get_digital(args, "train_mnist", resize=[2],colour=0,corruption=True)
            train_1 = get_digital(args, "train_usps", resize=[8],colour=0,corruption=False)
        elif args.corrupted_set == 1:
            train_0 = get_digital(args, "train_mnist", resize=[2],colour=0,corruption=False)
            train_1 = get_digital(args, "train_usps", resize=[8],colour=0,corruption=True)
        else:
            train_0 = get_digital(args, "train_mnist", resize=[2],colour=0,corruption=True)
            train_1 = get_digital(args, "train_usps", resize=[8],colour=0,corruption=True)
    else:
        train_0 = get_digital(args, "train_mnist", resize=[2],colour=0,corruption=False)
        train_1 = get_digital(args, "train_usps", resize=[8],colour=0,corruption=False)
        
    train_data = [train_0, train_1]
    return train_data
    
def mnist_Rmnist(args):
    """loading in the mnist Rmnist dataset with the correct dataset 
    corruption if specified."""
    if args.corrupted != 0:
        if  args.corrupted_set == 0:
            train_0 = get_digital(args, "train_mnist", resize=[2],colour=0,corruption=True)
            train_1 = get_digital(args, "train_Rmnist", resize=[2],colour=255,corruption=False)
        elif args.corrupted_set == 1:
            train_0 = get_digital(args, "train_mnist", resize=[2],colour=0,corruption=False)
            train_1 = get_digital(args, "train_Rmnist", resize=[2],colour=255,corruption=True)
        else:
            train_0 = get_digital(args, "train_mnist", resize=[2],colour=0,corruption=True)
            train_1 = get_digital(args, "train_Rmnist", resize=[2],colour=255,corruption=True)
    else:
        train_0 = get_digital(args, "train_mnist", resize=[2],colour=0,corruption=False)
        train_1 = get_digital(args, "train_Rmnist", resize=[2],colour=255,corruption=False)
        
    train_data = [train_0, train_1]
    return train_data

