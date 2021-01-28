"""
This contains the impelementation of our VAE network. It contains the encoder 
and decoder network with in the Encoder is the reparameterize trick.
"""
import numpy as np
from collections import OrderedDict
import torch
from torch import nn
from torch.nn import Parameter
from torchvision import models
from utils import init_weights
class Encoder(nn.Module):
    """the same encoder as in module.py, This is the VAE Training part to obtain
    the pretrained Encoder necesarry for the network in module.py to work"""
    def __init__(self):
        super(Encoder, self).__init__()

        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(32, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(16)

        self.fc1 = nn.Linear(8 * 8 * 16, 512)
        self.fc_bn1 = nn.BatchNorm1d(512)
        self.fc21 = nn.Linear(512, 64)
        self.fc22 = nn.Linear(512, 64)

        self.relu = nn.ReLU()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight)
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0.1)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, 1.0, 0.02)

    def reparameterize(self, mu, logvar):
        "reparameteriation trick to sample from a mu and sigma"
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = std.data.new(std.size()).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x):
        conv1 = self.relu(self.bn1(self.conv1(x)))
        conv2 = self.relu(self.bn2(self.conv2(conv1)))
        conv3 = self.relu(self.bn3(self.conv3(conv2)))
        conv4 = self.relu(self.bn4(self.conv4(conv3))).view(-1, 8 * 8 * 16)
        fc1 = self.relu(self.fc_bn1(self.fc1(conv4)))
        mu, logvar = self.fc21(fc1), self.fc22(fc1)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def get_parameters(self):
        return [{"params": self.parameters(), "lr_mult": 1}]

class Decoder(nn.Module):
        """a matching decoder network with the same shapes and layers 
        as the encoder so the VAE can be trained"""
        def __init__(self):
            super(Decoder, self).__init__()
            self.lin = nn.Linear(64, 32*16)
            act_fn = nn.ReLU
            self.net = nn.Sequential(
            nn.ConvTranspose2d(2*16, 2*16, kernel_size=3, output_padding=1, padding=1, stride=2), # 4x4 => 8x8
            act_fn(),
            nn.Conv2d(2*16, 2*16, kernel_size=3, padding=1),
            act_fn(),
            nn.ConvTranspose2d(2*16, 16, kernel_size=3, output_padding=1, padding=1, stride=2), # 8x8 => 16x16
            act_fn(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            act_fn(),
            nn.ConvTranspose2d(16,1, kernel_size=3, output_padding=1, padding=1, stride=2), # 16x16 => 32x32
            )

            self.relu = nn.ReLU()
    
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_uniform_(m.weight)
                if isinstance(m, nn.Linear):
                    nn.init.xavier_normal_(m.weight)
                    nn.init.constant_(m.bias, 0.1)
                if isinstance(m, nn.BatchNorm2d):
                    nn.init.normal_(m.weight, 1.0, 0.02)
    
        def forward(self, x):
            x = self.lin(x)
            x = x.reshape(x.shape[0],-1,4,4)
            x = self.net(x)
            return x
    
        def get_parameters(self):
            return [{"params": self.parameters(), "lr_mult": 1}]
