import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data.dataloader as dataloader
import torchvision.datasets as datasets
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset

import torchvision
import torchvision.transforms as tf

import numpy as np
import matplotlib.pyplot as plt
import time
from IPython.display import clear_output
from copy import deepcopy

## Dataloader ##

'''class STLData(Dataset):
    def __init__(self,trn_val_tst = 0, transform=None):
        data = np.load('hw3.npz')
        if trn_val_tst == 0:
            #trainloader
            self.images = data['arr_0']
            self.labels = data['arr_1']
        elif trn_val_tst == 1:
            #valloader
            self.images = data['arr_2']
            self.labels = data['arr_3']
        else:
            #testloader
            self.images = data['arr_4']
            self.labels = data['arr_5']
            
        self.images = np.float32(self.images)/1.0
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
   
        sample = self.images[idx,:]
        labels = self.labels[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample, labels'''

        # Model definition

class Q2_CNN(nn.Module):
    def __init__(self, nodes):
        super(Q2_CNN, self).__init__()
        
        self.nodes = nodes
        
        #conv-blk1
        self.conv1 = nn.Conv2d(nodes[0], nodes[1], 3, 2, 1)
        self.pool = nn.AdaptiveAvgPool2d(1) ## FIX
        self.fc1 = nn.Linear(192, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
                                               
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
                                               
        x = F.relu(self.conv7(x))
        x = F.relu(self.conv8(x))
        x = F.relu(self.conv9(x))
                                               
        x = F.relu(self.conv10(x))
        x = F.relu(self.conv11(x))
        x = F.relu(self.conv12(x))
                       
        x = self.pool(x)
        x = x.view(-1, self.nodes[4]) # Resize for FC layer
        x = self.fc1(x)
        return x

        ## Model training functions ##

def train_epoch(model, train_loader, criterion, optimizer, loss_logger, acc_logger):
    epoch_loss = 0
    epoch_acc = 0
    
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):   
        outputs = model(data.to(device))
        loss = criterion(outputs, target.to(device).long())
        
        optimizer.zero_grad()   
        loss.backward()
        optimizer.step()
        
        acc = calculate_accuracy(outputs, target.to(device))
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()

    loss_logger.append(epoch_loss / len(train_loader))
    acc_logger.append(epoch_acc / len(train_loader))
    
    return loss_logger, acc_logger

def test_model(model, test_loader, criterion, acc_logger):
    with torch.no_grad():
        model.eval()
        correct_predictions = 0
        total_predictions = 0
        for batch_idx, (data, target) in enumerate(test_loader):   
            outputs = model(data.to(device))
            
            #Calculate the accuracy of the model
            #you'll need to accumulate the accuracy over multiple steps
            
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == target.to(device)).sum().item()
            total_predictions += target.shape[0]
            
        acc_logger.append((correct_predictions/total_predictions)*100.0)
        return acc_logger

def calculate_accuracy(fx, y):
    preds = fx.max(1, keepdim=True)[1]
    correct = preds.eq(y.view_as(preds)).sum()
    acc = correct.float()/preds.shape[0]
    return acc

def evaluate(net, device, loader, Loss_fun, loss_logger, acc_logger):
    epoch_loss = 0
    epoch_acc = 0
    
    net.eval()
    
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            x = x.to(device)
            y = y.to(device)
            
            fx = net(x)
            loss = Loss_fun(fx, y.to(device).long())
            acc = calculate_accuracy(fx, y.to(device))
            
            #log the cumulative sum of the loss and acc
            epoch_loss += loss.item()
            epoch_acc += acc.item()
    
    loss_logger.append(epoch_loss / len(loader))
    acc_logger.append(epoch_acc / len(loader))
                
    #returns the logs for loss and accuracy per epoch     
    return loss_logger, acc_logger

def generate_confusion_matrix(model, device, loader): # Returns a 10x10 confusion matrix (indexed [target, predicted]) given some model and dataloader
    matrix = np.array([[0 for i in range(10)] for j in range(10)])
    model.eval()

    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            x = x.to(device)
            y = y.to(device)

            fx = model(x)

            for index, target in enumerate(y): # Loop over each sample
                # Get data about each sample
                target = int(target)

                fx_scores = list(fx[index]) # The output of the model given a single picture
                highest_score = max(fx_scores) # The highest score that the model gave
                guess = fx_scores.index(highest_score) # Which class the model guessed

                matrix[target, guess] += 1
    
    return matrix

transform = tf.Compose([tf.ToTensor(), tf.RandomErasing()])
## Data initialization

device = torch.device(0 if torch.cuda.is_available() else 'cpu')
batch_size = 100 

train_set = STLData(trn_val_tst=0, transform=transform) 
val_set = STLData(trn_val_tst=1, transform=transform) 
test_set = STLData(trn_val_tst=2, transform=transform) 

trainloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
valloader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False)
testloader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)

## Initializing model parameters

lr = 1e-4
n_epochs = 100

model = Q2_CNN([3, 32, 64, 128, 192]).to(device)
optimizer = optim.Adam(model.parameters(), lr)
criterion = nn.CrossEntropyLoss()

## Training model

train_loss = []
val_loss = []

train_acc = []
test_acc = []
val_acc = []

for i in range(n_epochs):
    train_loss, train_acc = train_epoch(model, trainloader, criterion, optimizer, train_loss, train_acc)
    test_acc = test_model(model, testloader, criterion, test_acc)
    val_loss, val_acc = evaluate(model, device, valloader, criterion, val_loss, val_acc)
    if (i+1)%10==0: print("Epoch: [{}/{}]".format(i+1, n_epochs))
    
print("Final accuracy of model: %.2f%%\n" % test_acc[-1])