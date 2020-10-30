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

