#from GAN import Generator, Discriminator
from dcgan import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader,Dataset
from datasets import GANdata
import glob,os
from torchvision.utils import save_image

device = torch.device('cuda' if is_cuda else 'cpu')

# img_size=(1,80,60)
batch_size=100
latent_dim=100

train_path = 'E:/연구실/지정맥/GAN/train'
valid_path = 'E:/연구실/지정맥/GAN/valid'
test_path = 'E:/연구실/지정맥/GAN/test'

train_dataset = GANdata(train_path)
trainset = DataLoader(train_dataset,
                     shuffle=True,
                     batch_size=batch_size)

noise = 100
generator = Generator().to(device)
discriminator = Discriminator().to(device)

model_path='E:/연구실/지정맥/GAN/save_dcgan/'
generator.load_state_dict(torch.load(model_path+"199generator.pt", map_location=device)) #
discriminator.load_state_dict((torch.load(model_path+'199discriminator.pt', map_location=device)))

with torch.no_grad():
    for i,test_imgs in enumerate(trainset):
        generator.eval()
        discriminator.eval()
        test_imgs = test_imgs.to(device)
        z = torch.randn(batch_size, noise).to(device)
        real_imgs = test_imgs
        fake_imgs = generator(z)
        for b in range(batch_size):
            save_image(fake_imgs[b],'dcgan_img_train/'+str(i) +'_'+str(b)+'.jpg')
