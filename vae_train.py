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

import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import make_grid, save_image
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from vae_utils import *
from vae_modules import Encoder, Decoder

import inspect
import sys
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir) 
from dataloader import *

# from bmnist import bmnist




class VAE(pl.LightningModule):

    def __init__(self, model_name, hidden_dims, num_filters, z_dim, lr):
        """
        PyTorch Lightning module that summarizes all components to train a VAE.
        Inputs:
            model_name - String denoting what encoder/decoder class to use.  Either 'MLP' or 'CNN'
            hidden_dims - List of hidden dimensionalities to use in the MLP layers of the encoder (decoder reversed)
            num_filters - Number of channels to use in a CNN encoder/decoder
            z_dim - Dimensionality of latent space
            lr - Learning rate to use for the optimizer
        """
        super().__init__()
        self.save_hyperparameters()

        self.encoder = Encoder()
        self.decoder = Decoder()


        self.criterion = torch.nn.BCEWithLogitsLoss(reduction="none")


    def forward(self, imgs):
        """
        The forward function calculates the VAE-loss for a given batch of images.
        Inputs:
            imgs - Batch of images of shape [B,C,H,W]
        Ouptuts:
            L_rec - The average reconstruction loss of the batch. Shape: single scalar
            L_reg - The average regularization loss (KLD) of the batch. Shape: single scalar
            bpd - The average bits per dimension metric of the batch.
                  This is also the loss we train on. Shape: single scalar
        """

        z, mean, log_std = self.encoder(imgs)

        # z = sample_reparameterize(mean, log_std)
        y = self.decoder(z)

        L_rec = self.criterion(y, imgs)
        B = L_rec.size()[0]
        L_rec = L_rec.view(B, -1)
        L_rec = torch.sum(L_rec, -1)

        L_reg = KLD(mean, log_std)

        elbo = (L_rec + L_reg)

        bpd = elbo_to_bpd(elbo, imgs.size())
        return L_rec, L_reg, bpd

    @torch.no_grad()
    def sample(self, bs):
        """
        Function for sampling a new batch of random images.
        Inputs:
            bs - Number of images to generate
        Outputs:
            x_samples - Sampled, binarized images with 0s and 1s. Shape: [B,C,H,W]
            x_mean - The sigmoid output of the decoder with continuous values
                     between 0 and 1 from which we obtain "x_samples".
                     Shape: [B,C,H,W]
        """
        z = torch.randn(bs, args.z_dim).to(self.decoder.device)

        y = self.decoder(z)

        x_mean = torch.sigmoid(y)

        x_samples = (x_mean > 0.5).float()

        return x_samples, x_mean

    def configure_optimizers(self):
        # Create optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer

    def training_step(self, batch, batch_idx):
        # Make use of the forward function, and add logging statements
        L_rec, L_reg, bpd = self.forward(batch[0])
        self.log("train_reconstruction_loss", L_rec, on_step=False, on_epoch=True)
        self.log("train_regularization_loss", L_reg, on_step=False, on_epoch=True)
        self.log("train_ELBO", L_rec + L_reg, on_step=False, on_epoch=True)
        self.log("train_bpd", bpd, on_step=False, on_epoch=True)

        return bpd

    def validation_step(self, batch, batch_idx):
        # Make use of the forward function, and add logging statements
        L_rec, L_reg, bpd = self.forward(batch[0])
        self.log("val_reconstruction_loss", L_rec)
        self.log("val_regularization_loss", L_reg)
        self.log("val_ELBO", L_rec + L_reg)
        self.log("val_bpd", bpd)

    def test_step(self, batch, batch_idx):
        # Make use of the forward function, and add logging statements
        L_rec, L_reg, bpd = self.forward(batch[0])
        self.log("test_bpd", bpd)


# class GenerateCallback(pl.Callback):

#     def __init__(self, bs=64, every_n_epochs=5, save_to_disk=False):
#         """
#         Inputs:
#             bs - Number of images to generate
#             every_n_epochs - Only save those images every N epochs (otherwise tensorboard gets quite large)
#             save_to_disk - If True, the samples and image means should be saved to disk as well.
#         """
#         super().__init__()
#         self.bs = bs
#         self.every_n_epochs = every_n_epochs
#         self.save_to_disk = save_to_disk

    # def on_epoch_end(self, trainer, pl_module):
    #     """
    #     This function is called after every epoch.
    #     Call the save_and_sample function every N epochs.
    #     """
    #     if (trainer.current_epoch+1) % self.every_n_epochs == 0:
    #         self.sample_and_save(trainer, pl_module, trainer.current_epoch+1)

    # def sample_and_save(self, trainer, pl_module, epoch):
    #     """
    #     Function that generates and saves samples from the VAE.
    #     The generated samples and mean images should be added to TensorBoard and,
    #     if self.save_to_disk is True, saved inside the logging directory.
    #     Inputs:
    #         trainer - The PyTorch Lightning "Trainer" object.
    #         pl_module - The VAE model that is currently being trained.
    #         epoch - The epoch number to use for TensorBoard logging and saving of the files.
    #     """
    #     # Hints:
    #     # - You can access the logging directory path via trainer.logger.log_dir, and
    #     # - You can access the tensorboard logger via trainer.logger.experiment
    #     # - Use the torchvision function "make_grid" to create a grid of multiple images
    #     # - Use the torchvision function "save_image" to save an image grid to disk

    #     samples, means = pl_module.sample(64)

    #     grid_samples = make_grid(samples).to('cpu')
    #     grid_means = make_grid(means).to('cpu')

    #     trainer.logger.experiment.add_image("samples_"+str(epoch), grid_samples)
    #     trainer.logger.experiment.add_image("means_"+str(epoch), grid_means)
        
    #     if self.save_to_disk:
    #         save_image(grid_samples,
    #                os.path.join(trainer.logger.log_dir, 'samples_' + str(epoch) + '.png'),
    #                normalize=False)
    #         save_image(grid_means,
    #                os.path.join(trainer.logger.log_dir, 'means_' + str(epoch) + '.png'),
    #                normalize=False)



def train_vae(args):
    """
    Function for training and testing a VAE model.
    Inputs:
        args - Namespace object from the argument parser
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    os.makedirs(args.log_dir, exist_ok=True)
    # train_loader, val_loader, test_loader = #DATALOADER
    data_loader = mnist_Rmnist(args)[1]
    train_loader, val_loader, test_loader = data_loader, data_loader, data_loader

    # Create a PyTorch Lightning trainer with the generation callback
    # gen_callback = GenerateCallback(save_to_disk=True)
    trainer = pl.Trainer(default_root_dir=args.log_dir,
                        #  checkpoint_callback=ModelCheckpoint(save_weights_only=True, mode="min", monitor="val_bpd"),
                         gpus=1 if torch.cuda.is_available() else 0,
                         max_epochs=args.epochs,
                        #  callbacks=[gen_callback],
                         progress_bar_refresh_rate=1 if args.progress_bar else 0)
    trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need

    # Create model
    pl.seed_everything(args.seed)  # To be reproducible
    model = VAE(model_name=args.model,
                hidden_dims=args.hidden_dims,
                num_filters=args.num_filters,
                z_dim=args.z_dim,
                lr=args.lr)

    # Training
    # gen_callback.sample_and_save(trainer, model, epoch=0)  # Initial sample
    trainer.fit(model, train_loader, val_loader)

    # Testing
    model = VAE.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
    test_result = trainer.test(model, test_dataloaders=test_loader, verbose=True)

    # Manifold generation
    if args.z_dim == 2:
        img_grid = visualize_manifold(model.decoder)
        save_image(img_grid,
                   os.path.join(trainer.logger.log_dir, 'vae_manifold.png'),
                   normalize=False)
    torch.save(model.encoder.state_dict(), "./save/encoder_Rmnist.pth")
    torch.save(model.decoder.state_dict(), "./save/decoder_Rmnist.pth")
    return test_result

# torch.autograd.set_detect_anomaly(True)
if __name__ == '__main__':
    # Feel free to add more argument parameters
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Model hyperparameters
    parser.add_argument('--model', default='MLP', type=str,
                        help='What model to use in the VAE',
                        choices=['MLP', 'CNN'])
    parser.add_argument('--z_dim', default=20, type=int,
                        help='Dimensionality of latent space')
    parser.add_argument('--hidden_dims', default=[512], type=int, nargs='+',
                        help='Hidden dimensionalities to use inside the network. To specify multiple, use " " to separate them. Example: "512 256"')
    parser.add_argument('--num_filters', default=32, type=int,
                        help='Number of channels/filters to use in the CNN encoder/decoder.')

    # Optimizer hyperparameters
    parser.add_argument('--lr', default=1e-3, type=float,
                        help='Learning rate to use')
    parser.add_argument('--bs', default=128, type=int,
                        help='Minibatch size')

    # Other hyperparameters
    parser.add_argument('--epochs', default=80, type=int,
                        help='Max number of epochs')
    parser.add_argument('--seed', default=42, type=int,
                        help='Seed to use for reproducing results')
    parser.add_argument('--num_workers', default=4, type=int,
                        help='Number of workers to use in the data loaders. To have a truly deterministic run, this has to be 0. ' + \
                             'For your assignment report, you can use multiple workers (e.g. 4) and do not have to set it to 0.')
    parser.add_argument('--log_dir', default='VAE_logs', type=str,
                        help='Directory where the PyTorch Lightning logs should be created.')
    parser.add_argument('--progress_bar', action='store_true',
                        help=('Use a progress bar indicator for interactive experimentation. '
                              'Not to be used in conjuction with SLURM jobs'))
    parser.add_argument('--corrupted', default=False,
                        help='Corrupted'
                        )
    args = parser.parse_args()

    train_vae(args)
