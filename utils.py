# @Author  : Peizhao Li
# @Contact : peizhaoli05@gmail.com
"""
here the weight initialisation,learning_rate schedulere and 
the cluster divergence regularization functions are implemented. 
"""
import random
import numpy as np
import torch
from torch import nn


def set_seed(seed):
    """
    setting the seeds to allow reproducability
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def init_weights(layer):
    """
    initialisation of the weitgths in the neural networks
    """
    layer_name = layer.__class__.__name__
    if layer_name.find("Conv2d") != -1 or layer_name.find("ConvTranspose2d") != -1:
        nn.init.kaiming_uniform_(layer.weight)
    elif layer_name.find("BatchNorm") != -1:
        nn.init.normal_(layer.weight, 1.0, 0.02)
    elif layer_name.find("Linear") != -1:
        nn.init.xavier_normal_(layer.weight)


def inv_lr_scheduler(optimizer, lr, iter, max_iter, gamma=10, power=0.75):
    learning_rate = lr * (1 + gamma * (float(iter) / float(max_iter))) ** (-power)
    i = 0
    for param_group in optimizer.param_groups:
        param_group["lr"] = learning_rate * param_group["lr_mult"]
        i += 1

    return optimizer


def target_distribution(batch):
    """
    Compute the target distribution p_ij, given the batch (q_ij), as in 3.1.3 Equation 3 of
    Xie/Girshick/Farhadi; this is used the KL-divergence loss function.

    :param batch: [batch size, number of clusters] Tensor of dtype float
    :return: [batch size, number of clusters] Tensor of dtype float
    """
    weight = (batch ** 2) / torch.sum(batch, 0)
    return (weight.t() / torch.sum(weight, 1)).t()


def aff(input):
    return torch.mm(input, torch.transpose(input, dim0=0, dim1=1))


def KL_divergence(output, target):
    'Wrapper function that calculates the KL-divergence'
    KLD = nn.KLDivLoss(reduction='sum')
    return KLD(output.log(), target)


def JS_divergence(output, target):
    "Calculates Jensen-Shannon divergence"

    KLD = nn.KLDivLoss(reduction='sum')
    M = 0.5 * (output + target)
    return 0.5 * KLD(output.log(), M) + 0.5 * KLD(target.log(), M)


def CS_divergence(output, target):
    "Calculates Cauchy-Schwarz divergence"

    numerator = torch.sum(output * target)
    denominator = torch.sqrt(torch.sum(output**2) * torch.sum(target**2))
    return -torch.log(numerator / denominator)

def K_means(args):
    #pooling defenition
    POOL = nn.MaxPool2d(4)
    args.bs = 512
    #load in the data
    data_loader = mnist_usps(args)
    #create k means mini batch object
    kmeans = sk.MiniBatchKMeans(10,max_iter=1000,batch_size=args.bs)
    len_image_0 = len(data_loader[0])
    len_image_1 = len(data_loader[1])
    #walk through the  whole dataset 1 time
    for step in range(int(60000/args.bs)):
        if step % len_image_0 == 0:
            iter_image_0 = iter(data_loader[0])
        if step % len_image_1 == 0:
            iter_image_1 = iter(data_loader[1])
        #load in the image and pool it down
        image_0, a = iter_image_0.__next__()
        image_1, _ = iter_image_1.__next__()
        image_0 = POOL(image_0)
        image_1 = POOL(image_1)
        #reshape in to 2d
        image_0 = image_0.reshape(args.bs,-1)
        image_1 = image_1.reshape(args.bs,-1)
        #run the kmeans on both datasets
        kmeans = kmeans.partial_fit(image_0)
        kmeans = kmeans.partial_fit(image_1)
        if step % 10 == 0:
            print(step, int(60000/args.bs))
            
    #save cluster centers in txt
    clusters = kmeans.cluster_centers_
    open("kmeans.txt", "w")
    file = open("kmeans.txt", "a")
    for i in range(len(clusters)):
        file.writelines([str(item)+" " for item in clusters[i]])
        file.writelines("\n")
        

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
