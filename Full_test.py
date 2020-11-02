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
from copy import deepcopy

# own libraries
from neuralNet import train_epoch, test_model, evaluate_model
from genalg import CNNGenAlgSolver

GPU_indx = 0
device = torch.device(GPU_indx if torch.cuda.is_available() else 'cpu')


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def plot_all(train_loss, test_loss, train_acc, test_acc):
    plt.figure(figsize = (12, 12))

    plt.subplot(2,2,1)
    plt.plot(train_loss)
    plt.title('Model Loss On Training Dataset Per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Training data loss')

    plt.subplot(2,2,2)
    plt.plot(test_loss)
    plt.title('Model Loss On Testing Dataset Per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Testing data loss')

    plt.subplot(2,2,3)
    plt.plot(train_acc)
    plt.title('Model Accuracy On Training Dataset')
    plt.xlabel('Epoch')
    plt.ylabel('Training data Accuracy')

    plt.subplot(2,2,4)
    plt.plot(test_acc)
    plt.title('Model Accuracy On Testing Dataset')
    plt.xlabel('Epoch')
    plt.ylabel('Testing data Accuracy')

    plt.show()


class CIFAR100Multiclass(Dataset):
    """
    Dataset containing nclasses classes, chosen randomly form input data
    """
    def __init__(self, data, classes, transforms=None):
        self.nclasses = len(classes)
        self.transforms = transforms
        self.size = self.nclasses * data.shape[1]
        self.permute = np.random.permutation(self.size)

        self.classes = classes
        self.images = np.float32(data[classes].reshape(self.size, 32, 32, 3).transpose(0, 3, 1, 2)[self.permute])/1.0
        self.labels = np.repeat(np.arange(self.nclasses), data.shape[1])[self.permute]

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = self.images[idx, :]
        label = self.labels[idx]
        if self.transforms:
            sample = self.transforms(sample)
        return sample, label


class SmallestModel(nn.Module):
    def __init__(self, nkernels, nclasses):
        super(SmallestModel, self).__init__()
        self.nkernels = nkernels
        self.nclasses = nclasses
        self.kernel_size = 5

        self.conv1 = nn.Conv2d(3, nkernels, kernel_size=self.kernel_size, padding=2, bias=False)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(nkernels, nkernels//2, kernel_size=5)
        self.gap = nn.AvgPool2d(12)
        self.fc = nn.Linear(nkernels//2, nclasses)

    def init_genome(self, data):
        self.conv1.weight = torch.nn.Parameter(data)

    def get_genome(self):
        return self.conv1.weight.data

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = torch.squeeze(self.gap(x))
        x = self.fc(x)
        return x

#   Rough work

# Testing code, delete later

# Loading data
metadata = unpickle('cifar-100-python/meta')
legend = metadata[b'fine_label_names']
train_data = unpickle('cifar-100-python/train_sort')
test_data = unpickle('cifar-100-python/test_sort')

the_nclasses = 4
the_classes = np.array(random.sample(range(100), the_nclasses))

train_set = CIFAR100Multiclass(train_data[:, :400], classes=the_classes)
test_set = CIFAR100Multiclass(test_data, classes=the_classes)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=50)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=100)



# Initializing network parameters

loss_func = nn.CrossEntropyLoss()
lr = 1e-3
nepochs = 30
optimizer = optim.Adam(bleh_network.parameters(), lr)

solver = CNNGenAlgSolver(
    pop_size=10, # population size (number of models)
    max_gen=500, # maximum number of generations
    mutation_rate=0.05, # mutation rate to apply to the population
    selection_rate=0.5, # percentage of the population to select for mating
    selection_strategy="roulette_wheel", # strategy to use for selection. see below for more details
    model=Model,
    num_channels=3,
    train_set=train_set,
    test_set=test_set,
    device=device,
    loss_function=loss_function,
    optimizer=optim.Adam,
    learning_rate=lr
)

solver.solve()

# end result is in solver.population
