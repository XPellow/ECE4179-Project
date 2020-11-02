import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data.dataloader as dataloader
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset

import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import numpy as np
import matplotlib.pyplot as plt
import time
from IPython.display import clear_output
from copy import deepcopy

    ## Dataloader ##
    
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True) #, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False) #, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    ## Model Definition ##

class Model(nn.Module):`
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 5, 2, 0) # In channels, out channels, kernel size, stride, padding
        self.pool = nn.MaxPool2d(2, 2) # Kernel size, stride, padding
        self.conv2 = nn.Conv2d(64, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x