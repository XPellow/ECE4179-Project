import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data.dataloader as dataloader
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset

import torchvision
import torchvision.transforms as tf
import torchvision.datasets as datasets

import numpy as np
import matplotlib.pyplot as plt
import time
from IPython.display import clear_output
from copy import deepcopy
from random import random


class Model(nn.Module):
    def __init__(self, nkernels, nclasses):
        super(Model, self).__init__()
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


def train_epoch(model, loader, loss_func, optimizer, device, loss_logger=[]):
    """
    Trains a given model, returning the loss log, and it's overall prediction %
    """
    model.train()
    correct = 0
    total = 0
    for i, (data, target) in enumerate(loader):
        target = target.long()
        output = model(data.to(device))
        _, predicted = torch.max(output, 1)
        correct += (predicted == target.to(device)).sum().item()
        total += target.shape[0]
        loss = loss_func(output, target.to(device))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_logger.append(loss.item())

    return loss_logger, (correct / total) * 100.0


def test_model(model, loader, loss_func, device, loss_logger=[]):
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for data, target in loader:
            target = target.long()
            output = model(data.to(device))
            _, predicted = torch.max(output, 1)
            correct += (predicted == target.to(device)).sum().item()
            total += target.shape[0]

            loss = loss_func(output, target.to(device))
            loss_logger.append(loss.item())

        return loss_logger, (correct / total) * 100.0


def evaluate_model(model, device, loader):
    """
    Returns the current accuracy of a given model.
    """
    epoch_acc = 0
    
    model.eval()
    
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            x = x.to(device)
            y = y.to(device)
            
            fx = model(x)

            predictions = fx.max(1, keepdim=True)[1]
            correct = predictions.eq(y.view_as(preds)).sum()
            acc = correct.float()/preds.shape[0]

            #log the cumulative sum of the loss and acc
            epoch_acc += acc.item()
    
    model_acc = epoch_acc / len(loader)
  
    return model_acc


def full_train(model, n_epochs, train_loader, test_loader, loss_func, optimizer, device, freeze): ## TODO - implement freezing of first layer
    """
    Train a given 'model' for 'n_epochs'
    """
    train_loss = []
    test_loss = []
    train_acc = []
    test_acc = []

    for i in range(n_epochs):
        #print("Epoch: [%d/%d]" % (i + 1, nepochs))

        epoch_loss, acc = train_epoch(model, train_loader, loss_func, optimizer, device)
        train_loss.append(sum(epoch_loss) / len(epoch_loss))
        train_acc.append(acc)

        _, acc = test_model(model, test_loader, loss_func, device)
        test_loss.append(sum(epoch_loss) / len(epoch_loss))
        test_acc.append(acc)
    return train_loss, test_loss, train_acc, test_acc


def freeze_layer(model, freeze=True, invert=False):
    """
    If 'invert' is true, the model's first layer will flip between frozen and unfrozen.
    Otherwise, freeze depending on 'freeze'
    """
    if invert:
        model.conv1.weight.requires_grad = not model.conv1.weight.requires_grad
    else:
        model.conv1.weight.requires_grad = not freeze # did you know that '~' is retarded

