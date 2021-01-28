################################################################################
# MIT License
#
# Copyright (c) 2020 Phillip Lippe
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course | Fall 2020
# Date Created: 2020-11-22
################################################################################
"""
These are all our extra functions necesarry for our vae
"""
import torch
from torchvision.utils import make_grid
import numpy as np
from scipy.stats import norm

def KLD(mean, log_std):
    """
    Calculates the Kullback-Leibler divergence of given distributions to unit Gaussians over the last dimension.
    See Section 1.3 for the formula.
    Inputs:
        mean - Tensor of arbitrary shape and range, denoting the mean of the distributions.
        log_std - Tensor of arbitrary shape and range, denoting the log standard deviation of the distributions.
    Outputs:
        KLD - Tensor with one less dimension than mean and log_std (summed over last dimension).
              The values represent the Kullback-Leibler divergence to unit Gaussians.
    """
    std = torch.exp(log_std)
    var = torch.square(std)
    log_var = torch.log(var)

    KLD = (var + torch.square(mean) - 1 - log_var) / 2
    KLD = torch.sum(KLD, -1)
    return KLD

def elbo_to_bpd(elbo, img_shape):
    """
    Converts the summed negative log likelihood given by the ELBO into the bits per dimension score.
    Inputs:
        elbo - Tensor of shape [batch_size]
        img_shape - Shape of the input images, representing [batch, channels, height, width]
    Outputs:
        bpd - The negative log likelihood in bits per dimension for the given image.
    """
    nll = torch.mean(elbo)
    prod = 1 / (np.prod(img_shape[1:]))
    bpd = (nll * prod) / np.log(2)
    return bpd
