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
    KERNEL_SIZE = 5

    def __init__(self, nkernels, nclasses, n_mut1=0, n_mut2=0):
        """
        Creates small model with some number of kernels in the first convolutional layer, and some number of output
        classes. Type 1 mutation refers to when a kernel is unfrozen and allowed to learn at a decreased rate, type 2
        mutation refers to when a kernel is randomly initialized and then allowed to learn.
        
        :param nkernels: number of kernels in the first layer of the network
        :param nclasses: number of classes output by the network
        :param n_mut1: number of kernels that undergo type 1 mutation
        :param n_mut2: number of kernels that undergo type 2 mutation
        """
        super(Model, self).__init__()
        self.nkernels = nkernels
        self.nclasses = nclasses
        self.mut1len = n_mut1  # number of kernels that are given type 1 mutation
        self.mut2len = n_mut2  # number of kernels that are given type 2 mutation
        self.nomutlen = nkernels - n_mut1 - n_mut2
        self.ingenlen = nkernels-n_mut2  # Length of genome input

        # Specify genome layer
        if n_mut1:
            self.genome_mut1 = nn.Conv2d(3, n_mut1, kernel_size=self.KERNEL_SIZE, padding=2, bias=False)
        if n_mut2:
            self.genome_mut2 = nn.Conv2d(3, n_mut2, kernel_size=self.KERNEL_SIZE, padding=2, bias=False)
        self.genome_nomut = nn.Conv2d(3, self.nomutlen, kernel_size=self.KERNEL_SIZE, padding=2, bias=False)

        # Specify other layers
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(nkernels, nkernels//2, kernel_size=5)
        self.gap = nn.AvgPool2d(12)
        self.fc = nn.Linear(nkernels//2, nclasses)

    def init_genome(self, data):
        """
        Initializes the first layer specified by a given genome, genome data input should be of length self.ingenlen
        :param data: Genome data as torch tensor
        :return: None
        """
        with torch.no_grad():
            if data.shape[0] > self.ingenlen:
                raise Exception("Don't try to give the model downs syndrome.")
            elif self.mut1len:
                self.genome_mut1.weight = torch.nn.Parameter(data[:self.mut1len])
                self.genome_nomut.weight = torch.nn.Parameter(data[self.mut1len:])
            else:
                self.genome_nomut.weight = torch.nn.Parameter(data)

    def gen_optimizer(self, lr, mut_lr1, mut_lr2):
        """
        Generates an Adam optimizer for the network where the non-mutating kernels are frozen, the type 1 mutating
        kernels are given a reduced learning rate, and the type 2 mutating kernels are given a different specified lr.

        :param lr: Default learning rate of the network
        :param mut_lr1: Rate at which type 1 mutating kernels learn.
        :param mut_lr2: Rate at which type 2 mutating kernels learn.
        :return: Adam optimizer for this network.
        """

        self.genome_nomut.weight.requires_grad = False
        optimdata = [{'params': [self.conv2.weight, self.conv2.bias, self.fc.weight, self.fc.bias]}]
        if self.mut1len: optimdata.append({'params': self.genome_mut1.weight, 'lr': mut_lr1})
        if self.mut2len: optimdata.append({'params': self.genome_mut2.weight, 'lr': mut_lr2})
        return optim.Adam(optimdata, lr=lr)

    def get_genome(self):

        weights = [self.genome_nomut.weight.data]
        if self.mut1len: weights.append(self.genome_mut1.weight.data)
        if self.mut2len: weights.append(self.genome_mut2.weight.data)
        return torch.cat(weights, 0)

    def forward(self, x):
        layer1comp = [self.genome_nomut(x)]
        if self.mut1len: layer1comp.append(self.genome_mut1(x))
        if self.mut2len: layer1comp.append(self.genome_mut2(x))
        x = F.relu(torch.cat(layer1comp, 1))
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


def full_train(model, n_epochs, train_loader, test_loader, loss_func, optimizer, device): ## TODO - implement freezing of first layer
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

