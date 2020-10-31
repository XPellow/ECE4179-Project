
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

GPU_indx = 0
device = torch.device(GPU_indx if torch.cuda.is_available() else 'cpu')


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


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


# Models

class SmallModel(nn.Module):
    def __init__(self, nkernels, nclasses):
        super(SmallModel, self).__init__()
        self.nkernels = nkernels
        self.nclasses = nclasses
        self.kernel_size = 5

        self.conv1 = nn.Conv2d(3, nkernels, kernel_size=kernel_size, padding=2, bias=False)
        self.conv2 = nn.Conv2d(nkernels, 2*nkernels, kernel_size=5)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv3 = nn.Conv2d(2*nkernels, 2*nkernels, kernel_size=5)
        self.gap = nn.AvgPool2d(10)
        self.fc = nn.Linear(2*nkernels, nclasses)

    def init_genome(self, data):
        self.conv1.weight = torch.nn.Parameter(data)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool1(x)
        x = F.relu(self.conv3(x))
        x = torch.squeeze(self.gap(x))
        x = self.fc(x)
        return x


class SmallerModel(nn.Module):
    def __init__(self, nkernels, nclasses):
        super(SmallerModel, self).__init__()
        self.nkernels = nkernels
        self.nclasses = nclasses
        self.kernel_size = 5

        self.conv1 = nn.Conv2d(3, nkernels, kernel_size=kernel_size, padding=2, bias=False)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(nkernels, 2*nkernels, kernel_size=5)
        self.gap = nn.AvgPool2d(12)
        self.fc = nn.Linear(2*nkernels, nclasses)

    def init_genome(self, data):
        self.conv1.weight = torch.nn.Parameter(data)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = torch.squeeze(self.gap(x))
        x = self.fc(x)
        return x


class SmallestModel(nn.Module):
    def __init__(self, nkernels, nclasses):
        super(SmallestModel, self).__init__()
        self.nkernels = nkernels
        self.nclasses = nclasses
        self.kernel_size = 5

        self.conv1 = nn.Conv2d(3, nkernels, kernel_size=kernel_size, padding=2, bias=False)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(nkernels, nkernels//2, kernel_size=5)
        self.gap = nn.AvgPool2d(12)
        self.fc = nn.Linear(nkernels//2, nclasses)

    def init_genome(self, data):
        self.conv1.weight = torch.nn.Parameter(data)

    def genome(self):
        return self.conv1.weight.data

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = torch.squeeze(self.gap(x))
        x = self.fc(x)
        return x



#
#  Crap i copied from other projects
#


def train_epoch(model, loader, loss_func, optimizer, loss_logger):
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


def test_model(model, loader, loss_func, loss_logger):
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


# Rough work

# Testing code, delete later
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

bleh_network = SmallestModel(32, the_nclasses).to(device)

loss_func = nn.CrossEntropyLoss()
lr = 1e-3
nepochs = 50
optimizer = optim.Adam(bleh_network.parameters(), lr)

train_loss, train_acc = [], []
test_loss, test_acc = [], []

for i in range(nepochs):
    print("Epoch: [%d/%d]" % (i + 1, nepochs))

    epoch_loss = []
    epoch_loss, acc = train_epoch(bleh_network, train_loader, loss_func, optimizer, epoch_loss)
    train_loss.append(sum(epoch_loss) / len(epoch_loss))
    train_acc.append(acc)

    epoch_loss = []
    _, acc = test_model(bleh_network, test_loader, loss_func, epoch_loss)
    test_loss.append(sum(epoch_loss) / len(epoch_loss))
    test_acc.append(acc)


plt.figure(figsize = (16, 3))

plt.subplot(1,2,1)
plt.plot(train_loss)
plt.title('Model Loss On Training Dataset Per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Training data loss')

plt.subplot(1,2,2)
plt.plot(test_loss)
plt.title('Model Loss On Testing Dataset Per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Testing data loss')

plt.show()

plt.figure(figsize = (16, 3))

plt.subplot(1,2,1)
plt.plot(train_acc)
plt.title('Model Accuracy On Training Dataset')
plt.xlabel('Epoch')
plt.ylabel('Training data Accuracy')

plt.subplot(1,2,2)
plt.plot(test_acc)
plt.title('Model Accuracy On Testing Dataset')
plt.xlabel('Epoch')
plt.ylabel('Testing data Accuracy')

plt.show()

# Now train another network with the same first layer as the parent network, frozen

child_network = SmallestModel(32, the_nclasses).to(device)
child_network.init_genome(bleh_network.genome())
child_network.conv1.weight.requires_grad = False
print(child_network.genome()[0])
print(child_network.fc.weight.data)

the_classes2 = np.array(random.sample(range(100), the_nclasses))

train_set2 = CIFAR100Multiclass(train_data[:, :400], classes=the_classes2)
test_set2 = CIFAR100Multiclass(test_data, classes=the_classes2)
train_loader2 = torch.utils.data.DataLoader(train_set, batch_size=50)
test_loader2 = torch.utils.data.DataLoader(test_set, batch_size=100)

optimizer2 = optim.Adam(filter(lambda p: p.requires_grad, child_network.parameters()), lr)

train_loss2, train_acc2 = [], []
test_loss2, test_acc2 = [], []

for i in range(nepochs):
    print("Epoch: [%d/%d]" % (i + 1, nepochs))

    epoch_loss = []
    epoch_loss, acc = train_epoch(child_network, train_loader2, loss_func, optimizer2, epoch_loss)
    train_loss2.append(sum(epoch_loss) / len(epoch_loss))
    train_acc2.append(acc)

    epoch_loss = []
    _, acc = test_model(child_network, test_loader2, loss_func, epoch_loss)
    test_loss2.append(sum(epoch_loss) / len(epoch_loss))
    test_acc2.append(acc)

print(child_network.genome()[0])
print(child_network.fc.weight.data)

plt.figure(figsize = (16, 3))

plt.subplot(1,2,1)
plt.plot(train_loss2)
plt.title('Model Loss On Training Dataset Per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Training data loss')

plt.subplot(1,2,2)
plt.plot(test_loss2)
plt.title('Model Loss On Testing Dataset Per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Testing data loss')

plt.show()

plt.figure(figsize = (16, 3))

plt.subplot(1,2,1)
plt.plot(train_acc2)
plt.title('Model Accuracy On Training Dataset')
plt.xlabel('Epoch')
plt.ylabel('Training data Accuracy')

plt.subplot(1,2,2)
plt.plot(test_acc2)
plt.title('Model Accuracy On Testing Dataset')
plt.xlabel('Epoch')
plt.ylabel('Testing data Accuracy')

plt.show()