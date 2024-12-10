import time

import matplotlib.pyplot as plt

import torch
from torchvision import datasets
import torchvision.transforms.v2 as transforms

import models



ds_transform = transforms.Compose([
    transforms.ToImage(),
    transforms.ToDtype(torch.float32,scale=True)
])

ds_train = datasets.FashionMNIST(
    root='data',
    train=True,
    download=True,
    transform=ds_transform
)

batch_size = 64
dataloader_train = torch.utils.data.DataLoader(
    ds_train,
    batch_size=batch_size,
    shuffle=True
)
dataloader_test = torch.utils.data.DataLoader(
    ds_test,
    batch_size=batch_size
)


for image_batch, label_batch in dataloader_test:
    print(image_batch.shape)
    print(label_batch.shape)
    break