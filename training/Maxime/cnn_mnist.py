#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 16 19:13:57 2025

@author: Sdt. Maxime Hurtubise
"""

import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

num_classes = 10
num_epochs = 10
batch_size = 100
learning_rate = 5e-4
#beta = 0.9

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

# %% CNN
class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, 5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Linear(7*7*32, num_classes)
    
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.fc(out.flatten(start_dim=1))
        return out

model = CNN(num_classes).to(device)

# %% Parameters

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print('Number of parameters = {}'.format(count_parameters(model)))

# %% Training and Validation

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

num_steps = len(train_loader)

for epoch in range (num_epochs):
    for i, (images, labels) in enumerate (train_loader):
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        #Backward and Optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1)%100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{num_steps}], Loss [{loss.item():.4f}]')
            
print('Finished Training')

with torch.no_grad():
    num_correct = 0
    num_samples = 0
    for images, labels in test_loader:
        outputs = model(images).to(device)
        labels = labels.to(device)
        
        _, prediction = torch.max(outputs, 1)
        num_samples += labels.shape[0]
        num_correct += (prediction == labels).sum().item()
        
    accuracy = 100.0 * num_correct / num_samples
    print(f'accuracy = {accuracy}%')
    







