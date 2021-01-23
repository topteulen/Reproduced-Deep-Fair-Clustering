# @Author  : Peizhao Li
# @Contact : peizhaoli05@gmail.com

from PIL import Image
import numpy as np

import torchvision.transforms.functional as TF
import random
import torch
from torch.utils import data
from torchvision import transforms

kwargs = {"shuffle": True, "num_workers": 0, "pin_memory": True, "drop_last": True}


class digital(data.Dataset):
    # with size in 32x32
    def __init__(self, subset, transform=None):
        file_dir = "./data/{}.txt".format(subset)
        self.data_dir = open(file_dir).readlines()
        self.transform = transform

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

    def __init__(self, percent):
        self.percent = percent

    def __call__(self, img):
        img = np.array(img)
        base_color = img[0][0]
        length = len(img)
        for x in range(length):
            chance = np.random.rand(length)
            # if chance > self.percent:
            #     if base_color == img[x][y]:
            img[x][chance > self.percent] = 255-base_color

        img = Image.fromarray(img)
        return img




def get_digital(args, subset,resize=[],colour=0,corruption=False):
    if len(resize) != 0:
        if corruption == True:
            if 1 > args.corrupted > 0:
                transform = transforms.Compose([transforms.Pad(resize[0],colour),
                                                CorruptionTransform(args.corrupted),
                                                transforms.ToTensor(),
                                            ])
        else:
            transform = transforms.Compose([transforms.Pad(resize[0],colour),
                                            transforms.ToTensor(),
                                            ])

    else:
        transform = transforms.Compose([transforms.ToTensor(),
                                        # transforms.Normalize((0.5,), (0.5,))
                                        #transforms.Resize((resize[0],resize[1]))
                                        ])

    data = digital(subset, transform)
    data_loader = torch.utils.data.DataLoader(
        dataset=data,
        batch_size=args.bs,
        **kwargs
    )

    return data_loader


def mnist_usps(args):
    train_0 = get_digital(args, "train_mnist", resize=[2],colour=0,corruption=False)
    train_1 = get_digital(args, "train_usps", resize=[8],colour=0,corruption=False)
    train_data = [train_0, train_1]

    return train_data
    
def mnist_Rmnist(args):
    train_0 = get_digital(args, "train_mnist", resize=[2],colour=0)
    train_1 = get_digital(args, "train_Rmnist", resize=[2],colour=255)
    train_data = [train_0, train_1]

    return train_data

