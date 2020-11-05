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
from neuralNet import train_epoch, test_model, evaluate_model, Model
from genalg import CNNGenAlgSolver

GPU_indx = 0
device = torch.device(GPU_indx if torch.cuda.is_available() else 'cpu')

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def plot_all(train_loss, test_loss, train_acc, test_acc, gen):
    """
    Used for plotting a single generation
    """
    plt.figure(figsize = (12, 12))

    plt.subplot(2,2,1)
    plt.plot(train_loss)
    plt.title('Model Loss On Training Dataset Per Epoch. Generation={}'.format(gen))
    plt.xlabel('Epoch')
    plt.ylabel('Training data loss')

    plt.subplot(2,2,2)
    plt.plot(test_loss)
    plt.title('Model Loss On Testing Dataset Per Epoch. Generation={}'.format(gen))
    plt.xlabel('Epoch')
    plt.ylabel('Testing data loss')

    plt.subplot(2,2,3)
    plt.plot(train_acc)
    plt.title('Model Accuracy On Training Dataset. Generation={}'.format(gen))
    plt.xlabel('Epoch')
    plt.ylabel('Training data Accuracy')

    plt.subplot(2,2,4)
    plt.plot(test_acc)
    plt.title('Model Accuracy On Testing Dataset. Generation={}'.format(gen))
    plt.xlabel('Epoch')
    plt.ylabel('Testing data Accuracy')

    plt.show()


def plot_all_ontop(train_loss, test_loss, train_acc, test_acc, gen_idx):
    """
    Used for plotting multiple generations various graphics ontop of one another
    """
    legend = ["Generation {}".format(i) for i in gen_idx]

    plt.figure(figsize = (12, 12))

    plt.subplot(2,2,1)
    plt.plot(train_loss[gen_idx, :].T)
    plt.title('Model Loss On Training Dataset Per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Training data loss')
    plt.legend(legend)

    plt.subplot(2,2,2)
    plt.plot(test_loss[gen_idx, :].T)
    plt.title('Model Loss On Testing Dataset Per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Testing data loss')
    plt.legend(legend)

    plt.subplot(2,2,3)
    plt.plot(train_acc[gen_idx, :].T)
    plt.title('Model Accuracy On Training Dataset')
    plt.xlabel('Epoch')
    plt.ylabel('Training data Accuracy')
    plt.legend(legend)

    plt.subplot(2,2,4)
    plt.plot(test_acc[gen_idx, :].T)
    plt.title('Model Accuracy On Testing Dataset')
    plt.xlabel('Epoch')
    plt.ylabel('Testing data Accuracy')
    plt.legend(legend)

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


# Loading data
metadata = unpickle('cifar-100-python/meta')
legend = metadata[b'fine_label_names']
train_data = unpickle('cifar-100-python/train_sort')
test_data = unpickle('cifar-100-python/test_sort')

# User-chosen parameters

testing = False # If True, use predefined paramters for a quick check shit works
nclasses = 4
nloaders = 50
nkernels = 64
n_epochs = 50
pop_size = 12
pool_size = 5
max_gen = 10

lr = 1e-3
lr_mut1 = 1e-5
lr_mut2 = 1e-5

# Setting up multi-class loaders

train_loaders = []
test_loaders = []

for i in range(nloaders):
    new_class_indicies = np.array(random.sample(range(100), nclasses))
    new_train_set = CIFAR100Multiclass(train_data[:, :400], classes=new_class_indicies)
    new_test_set = CIFAR100Multiclass(test_data, classes=new_class_indicies)
    new_train_loader = torch.utils.data.DataLoader(new_train_set, batch_size=50)
    new_test_loader = torch.utils.data.DataLoader(new_test_set, batch_size=100)

    train_loaders.append(new_train_loader)
    test_loaders.append(new_test_loader)

train_loaders = np.array(train_loaders)
test_loaders = np.array(test_loaders)

# Initializing network parameters

loss_func = nn.CrossEntropyLoss()
optimizer = optim.Adam

if testing:
    solver = CNNGenAlgSolver(
        model=Model,
        pop_size=6, # population size (number of models)
        pool_size=3, # num of models chosen when creating the next generation
        max_gen=3, # maximum number of generations
        mutation_rate=0.05, # mutation rate to apply to the population
        num_channels=3,
        train_loaders=train_loaders,
        test_loaders=test_loaders,
        kernel_size=5,
        device=device,
        loss_function=loss_func,
        optimizer=optimizer,
        lr=lr,
        lr_mut1=lr_mut1,
        lr_mut2=lr_mut2,
        n_epochs=20,
        nkernels=4,
        nclasses=nclasses
    )
else:
    solver = CNNGenAlgSolver(
        model=Model,
        pop_size=pop_size, # population size (number of models)
        pool_size=pool_size, # num of models chosen when creating the next generation
        max_gen=max_gen, # maximum number of generations
        mutation_rate=0.05, # mutation rate to apply to the population
        num_channels=3,
        train_loaders=train_loaders,
        test_loaders=test_loaders,
        kernel_size=5,
        device=device,
        loss_function=loss_func,
        optimizer=optimizer,
        lr=lr,
        lr_mut1=lr_mut1,
        lr_mut2=lr_mut2,
        n_epochs=n_epochs,
        nkernels=nkernels,
        nclasses=nclasses
    )

solver.solve()

# Final population & all loggers in solver.xxx

ave_train_losses = np.average(solver.train_losses, axis=1)
ave_test_losses = np.average(solver.test_losses, axis=1)
ave_train_accs = np.average(solver.train_accs, axis=1)
ave_test_accs = np.average(solver.test_accs, axis=1)
print(solver.train_losses)
print("\n\n\n\n\n")
print(ave_train_losses)

if testing:
    plot_all_ontop(ave_train_losses, ave_test_losses, ave_train_accs, ave_test_accs, gen_idx=[0,1,2])
else:
    plot_all_ontop(ave_train_losses, ave_test_losses, ave_train_accs, ave_test_accs, gen_idx=range(max_gen+1))

# Plot a specific network
gen=2
plot_all(solver.train_losses[gen][0], 
         solver.test_losses[gen][0],
         solver.train_accs[gen][0],
         solver.test_accs[gen][0],
         gen=gen)

#plot_all(ave_train_losses, ave_test_losses, ave_train_accs, ave_test_accs, gen=10)