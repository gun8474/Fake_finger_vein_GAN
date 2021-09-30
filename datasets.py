# -*- coding: utf-8 -*-
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torchvision.utils as utils
# import torchvision.transforms as transforms
# import numpy as np
# from matplotlib import pyplot as plt
# import glob
# from PIL import Image
# from torchvision import transforms
# import os
#
# class GANdata():
#     def __init__(self):
#         self.dir = 'E:/지정맥/real_frame'
#         folders = os.listdir(self.dir)  # 1 ~ 20
#         self.label = []
#         self.width = 640
#         self.height = 480
#
#         file_names = [os.listdir(os.path.join(self.dir, f)) for f in folders]
#         self.file_name = sum(file_names, [])  # 4개씩나눠있던 파일들을 합침
#         for f in range(len(file_names)):
#             for i in range(len(file_names[f])):
#                 self.label.append(1)
#
#     def __len__(self):
#         return len(self.file_name)
#
#     def __getitem__(self, index):
#         transform = transforms.Compose([transforms.Resize((self.width, self.height)),
#                                         transforms.ToTensor()])
#
#         path = os.path.join(self.dir, self.file_name[index].split('_')[0], self.file_name[index])
#         print(path)
#
#         img = Image.open(path)
#         image = transform(img)
#
#         label = self.label[index]
#
#         return image, label

#-----------------------------------------------------------------------------------------------------------------------------------------------------#

# import os
# from PIL import Image
# from torchvision import transforms
# from torch.utils.data import Dataset
# import torchvision.transforms as transforms


# class GANdata(Dataset):
#     def __init__(self, path):
#         self.img_path = path
#         self.img_file = os.listdir(self.img_path)

#     def __len__(self):
#         print(len(self.img_file))
#         return len(self.img_file)

#     def __getitem__(self, index):
#         # transform = transforms.Compose([transforms.ToTensor()])
#         transform = transforms.Compose([transforms.ToTensor()])

#         image = os.path.join(self.img_path, self.img_file[index])
#         img = Image.open(image)
#         img = transform(img)
#         return img

#-----------------------------------------------------------------------------------------------------------------------------------------------------------#
import os
from PIL import Image
import torchvision.transforms as transforms


class GANdata():
    def __init__(self, path):
        self.img_path = path
        self.img_file = os.listdir(self.img_path)

    def __len__(self):
        return len(self.img_file)

    def __getitem__(self, index):
        transform = transforms.Compose([transforms.Resize((80,60)),
                                        transforms.ToTensor()])

        image = os.path.join(self.img_path, self.img_file[index])
        img = Image.open(image)
        img = transform(img)
        return img
# path = '/workspace/GAN/real_frame'  
# a = GANdata(path)
# print(a.__len__())
