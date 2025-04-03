import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
from collections import OrderedDict

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)


def shard_dataset(dataset, num_shards=10):
    indices = list(range(len(dataset)))
    np.random.shuffle(indices)
    shard_size = len(dataset) // num_shards
    shards = [Subset(dataset, indices[i * shard_size:(i + 1) * shard_size]) for i in range(num_shards)]
    return shards

shards = shard_dataset(trainset, num_shards=10)



class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 10)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.pool(self.conv1(x)))
        x = self.relu(self.pool(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def train_model_on_shard(model, shard, epochs=1, batch_size=64, lr=0.01):
    trainloader = DataLoader(shard, batch_size=batch_size, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    
    gradients = None

    for epoch in range(epochs):
        for inputs, labels in trainloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            if gradients is None:
                gradients = [param.grad.clone() for param in model.parameters()]
            else:
                for i, param in enumerate(model.parameters()):
                    gradients[i] += param.grad.clone()
        
    return gradients

def federated_sgd(gradients_list):
    avg_gradients = []

    num_clients = len(gradients_list)

    for i in range(len(gradients_list[0])):
        avg_grad = sum([gradients[i] for gradients in gradients_list]) / num_clients
        avg_gradients.append(avg_grad)

    return avg_gradients


models = [SimpleCNN() for _ in range(10)]
all_gradients = []

for i, shard in enumerate(shards):
    print(f"Training model {i+1} on shard {i+1}")
    model = models[i]
    model_gradients = train_model_on_shard(model, shard, epochs=1)
    all_gradients.append(model_gradients)

avg_gradients = federated_sgd(all_gradients)
global_model = SimpleCNN()
optimizer = optim.SGD(global_model.parameters(), lr=0.01, momentum=0.9)

with torch.no_grad():
    for param, avg_grad in zip(global_model.parameters(), avg_gradients):
        #param.grad = avg_grad
        param.data -= avg_grad * 0.01

    #optimizer.step()
    optimizer.zero_grad()


testloader = DataLoader(testset, batch_size=64, shuffle=False)
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in testloader:
        outputs = global_model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Accuracy : {100 * correct / total}%")
