# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 16:46:11 2020

@author: M
"""

import torch
# import numpy as np
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

input_size = 784
hidden_size = 500
num_classes = 10
num_epoches = 5
batch_size = 100
learning_rate = 0.001

train_dataset = torchvision.datasets.MNIST(root='E:/pro/data',
                                           train=True,
                                           transform=transforms.ToTensor(),
                                           download = True)

test_dataset = torchvision.datasets.MNIST(root='E:/pro/data',
                                          train=False,
                                          transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size = batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

class NeuralNet(nn.Module):
    def __init__(self,input_size,hidden_size,num_classes):
        super(NeuralNet,self).__init__()
        self.fc1 = nn.Linear(input_size,hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size,num_classes)
        
    def forward(self,x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

model = NeuralNet(input_size,hidden_size,num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)

total_step = len(train_loader)
for epoch in range(num_epoches):
    for i ,(images,labels) in enumerate(train_loader):
        images = images.reshape(-1,28*28).to(device)
        labels = labels.to(device)
        
        outputs = model(images)
        loss = criterion(outputs,labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 100 == 0:
            print('Epoch [{}/{}], Step[{}/{}],Loss:{:.4f}'.format(epoch+1,num_epoches,i+1,total_step,loss.item()))

with torch.no_grad():
    correct = 0
    total = 0
    for images,lables in test_loader:
        images = images.reshape(-1,28*28).to(device)
        lables = lables.to(device)
        outputs = model(images)
        _,predicted = torch.max(outputs.data,1)
        total += lables.size(0)
        correct += (predicted == labels).sum().item() 

    print("Accurancy:%.4f" % (100*correct/total))
    
torch.save(model.state_dict(),'model.ckpt')
        