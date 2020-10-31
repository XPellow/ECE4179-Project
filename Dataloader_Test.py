import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data.dataloader as dataloader
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset

import torchvision
import torchvision.transforms as tf
import torchvision.datasets as datasets

import pickle
import random

import numpy as np
import matplotlib.pyplot as plt
import time
from IPython.display import clear_output
from copy import deepcopy


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


class CIFAR100Multiclass(Dataset):
    """
    Dataset containing nclasses classes, chosen randomly form input data
    """
    def __init__(self, data, nclasses=100, transforms=None):
        self.nclasses = nclasses
        self.transforms = transforms
        self.size = nclasses * data.shape[1]
        self.permute = np.random.permutation(self.size)

        self.classes = random.sample(range(100), nclasses)
        self.images = np.float32(data[self.classes].reshape(self.size, 32, 32, 3)[self.permute])/1.0
        self.labels = np.repeat(np.arange(nclasses), data.shape[1])[self.permute]

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = self.images[idx, :]
        labels = self.labels[idx]
        if self.transforms:
            sample = self.transforms(sample)
        return sample, labels


# Testing code, delete later
metadata = unpickle('cifar-100-python/meta')
legend = metadata[b'fine_label_names']
train_data = unpickle('cifar-100-python/train_sort')
test_data = unpickle('cifar-100-python/test_sort')

bleh_set = CIFAR100Multiclass(train_data, nclasses=2)
bleh_loader = torch.utils.data.DataLoader(bleh_set, batch_size=500)
image_batch, labels = next(iter(bleh_loader))
for tmpC1 in range(8):
    img = image_batch[tmpC1].numpy()
    plt.subplot(2, 4, tmpC1+1)
    plt.title(labels[tmpC1].item())
    plt.imshow(img/255.0)
plt.show()


