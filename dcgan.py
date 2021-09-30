import argparse
import os
import numpy as np
import math
import torch

import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F

from datasets import *
from earlystopping import EarlyStopping

is_cuda = torch.cuda.is_available()
print(is_cuda)
device = torch.device('cuda' if is_cuda else 'cpu')
print(device)

# hyperparameters
batch_size = 100
epochs = 200
noise = 100
hidden = 256
img_size = 64
latent_dim = 100


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        # self.latent_dims = latent_dims
        self.init_size = img_size // 4
        self.l1 = nn.Sequential(nn.Linear(latent_dim, 128 * self.init_size ** 2))

        # self.l1 = nn.Linear(latent_dims, 128)
        self.batch1 = nn.BatchNorm2d(128)
        self.up1 = nn.Upsample(scale_factor=2)
        self.conv1 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
        self.batch2 = nn.BatchNorm2d(128, 0.8)
        self.leak1 = nn.LeakyReLU(0.2, inplace=True)
        self.up2 = nn.Upsample(scale_factor=2)
        self.conv2 = nn.Conv2d(128, 64, 3, stride=1, padding=1)
        self.batch3 = nn.BatchNorm2d(64, 0.8)
        self.leak2 =  nn.LeakyReLU(0.2, inplace=True)
        self.conv3 = nn.Conv2d(64, 3, 3, stride=1, padding=1)
        nn.tanh = nn.Tanh()
        # self.conv_blocks = nn.Sequential(
        #     nn.BatchNorm2d(128),
        #     nn.Upsample(scale_factor=2),
        #     nn.Conv2d(128, 128, 3, stride=1, padding=1),
        #     nn.BatchNorm2d(128, 0.8),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Upsample(scale_factor=2),
        #     nn.Conv2d(128, 64, 3, stride=1, padding=1),
        #     nn.BatchNorm2d(64, 0.8),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Conv2d(64, 3, 3, stride=1, padding=1),
        #     nn.Tanh(),
        # )

    def forward(self, z):
        out = self.l1(z).to(device)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size).to(device)
        out = self.batch1(out)
        out = self.up1(out)
        out = self.conv1(out)
        out = self.batch2(out)
        out = self.leak1(out)
        out = self.up2(out)
        out = self.conv2(out)
        out = self.batch3(out)
        out = self.leak2(out)
        out = self.conv3(out)

        # out = out.view(out.shape[0], 128, self.init_size, self.init_size).to(device)
        # print('generator', out.shape)
        # img = self.conv_blocks(out)
        return out


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(3, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        ds_size = img_size // 2 ** 4
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size * ds_size, 1), nn.Sigmoid())

    def forward(self, img):
        print('img', img.shape)
        out = self.model(img)
        print('bbbbbbbbbbbbbbb', out.shape)
        out = out.view(out.shape[0], -1)
        print('aaaaaaaaaaaaaaa', out.shape)
        out = self.adv_layer(out)
        return out

g = Generator()
z = torch.randn(100, noise)
g(z)

