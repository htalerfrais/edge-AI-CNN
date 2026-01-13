#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 16 21:38:43 2025

@author: Sdt. Maxime Hurtubise
"""

import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import math

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

num_classes = 10
num_epochs = 10
batch_size = 100
learning_rate = 0.01
beta = 0.9
H = 100

train_dataset = torchvision.datasets.MNIST(root = './data', train = True, transform=transforms.ToTensor(), download=True)
test_dataset = torchvision.datasets.MNIST(root = './data', train = False, transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

examples = iter(train_loader)
samples, labels = next(examples)
print(samples.shape, labels.shape)

# %% Show samples
#for i in range (10):
    #plt.subplot(2, 5, i+1)
    #plt.imshow(samples[i][0], cmap='gray')
#plt.show()

# New size = (input_width - filter_width + 2*padding)/stride + 1

# %% MLP
class MLP(nn.Module):
    def __init__(self, H):
        super(MLP, self).__init__()
        
        self.C = 10
        self.D = 784
        self.H = H
        
        self.fc1 = nn.Linear(self.D, self.H) 
        self.relu = nn.ReLU()
        self.fc3 = nn.Linear(self.H, self.C)  
        
        #init parameters
        with torch.no_grad():
            self.fc1.weight.uniform_(-math.sqrt(6./self.D), math.sqrt(6./self.D))
            self.fc1.bias.uniform_(-1./math.sqrt(self.D), 1./math.sqrt(self.D))
            self.fc3.weight.uniform_(-math.sqrt(6./self.H),math.sqrt(6./self.H))
            self.fc3.bias.uniform_(-1./math.sqrt(self.H),1./math.sqrt(self.H))
        
    def forward(self,X):
    
        X1 = self.fc1(X) #NxH
        X2 = self.relu(X1) #NxH
        S = self.fc3(X2) #NxC
    
        return S
    
    
model = MLP(H).to(device)

# %% Parameters

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print('Number of parameters = {}'.format(count_parameters(model)))

# %% Training and Validation

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=beta)

num_steps = len(train_loader)

for epoch in range (num_epochs):
    for i, (images, labels) in enumerate (train_loader):
        # Forward pass
        images = images.view(-1, 28*28).to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        #Backward and Optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1)%100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{num_steps}], Loss [{loss.item():.4f}]')
            
print('Finished Training')

model.eval()
with torch.no_grad():
    num_correct = 0
    num_samples = 0
    for images, labels in test_loader:
        images = images.view(-1, 28*28).to(device)
        outputs = model(images).to(device)
        labels = labels.to(device)
        
        _, prediction = torch.max(outputs, 1)
        num_samples += labels.shape[0]
        num_correct += (prediction == labels).sum().item()
        
    accuracy = 100.0 * num_correct / num_samples
    print(f'accuracy = {accuracy}%')


