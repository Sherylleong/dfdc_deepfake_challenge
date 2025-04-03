import os
import numpy as np
import pandas as pd
import matplotlib
import torch
import matplotlib.pyplot as plt
from torchvision.datasets import ImageFolder
from PIL import Image
from tqdm import tqdm
from efficientnet_pytorch import EfficientNet

#from albumentations.pytorch import ToTensor 
from torch.utils.data import Dataset, random_split
from torch.utils.data import DataLoader, ConcatDataset
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import sys
import sklearn

import cv2
import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
from torchvision import models, transforms
from torch.utils.data import WeightedRandomSampler
 
import matplotlib.pyplot as plt

deepfake_types = ['Deepfakes', 'Face2Face', 'FaceShifter','FaceSwap', 'NeuralTextures' ]
IMG_SIZE = 224
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
BATCH_SIZE = 64
device = 'cuda'

LR = 0.001
loss_fn = torch.nn.BCEWithLogitsLoss()
class ImageTransform:
    def __init__(self, size, mean, std):
        self.data_transform = transforms.Compose([
                transforms.Resize((size, size), interpolation=Image.BILINEAR),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])

    def __call__(self, img):
        return self.data_transform(img)

# load and prep data

def get_dataset_size(DEEPFAKE_TYPE):
    transformer = ImageTransform(IMG_SIZE, mean, std)

    TRAIN_FOLDER = fr'D:\FF\crops\{DEEPFAKE_TYPE}\train' 
    VAL_FOLDER = fr'D:\FF\crops\{DEEPFAKE_TYPE}\val' 
    TEST_FOLDER = fr'D:\FF\crops\{DEEPFAKE_TYPE}\test' 

    train_ds = ImageFolder(root=TRAIN_FOLDER, transform=transformer)

    return len(train_ds)

def load_dataset(DEEPFAKE_TYPE):
    transformer = ImageTransform(IMG_SIZE, mean, std)

    TRAIN_FOLDER = fr'D:\FF\crops\{DEEPFAKE_TYPE}\train' 
    VAL_FOLDER = fr'D:\FF\crops\{DEEPFAKE_TYPE}\val' 
    TEST_FOLDER = fr'D:\FF\crops\{DEEPFAKE_TYPE}\test' 

    train_ds = ImageFolder(root=TRAIN_FOLDER, transform=transformer)
    val_ds = ImageFolder(root=VAL_FOLDER, transform=transformer)
    test_ds = ImageFolder(root=TEST_FOLDER, transform=transformer)

    train_loader = DataLoader(train_ds, shuffle=True, batch_size=64, num_workers=1)
    val_loader = DataLoader(val_ds, shuffle=True, batch_size=64, num_workers=8)
    test_loader = DataLoader(test_ds, shuffle=False, batch_size=64, num_workers=8)
    return train_loader, val_loader, test_loader

def load_datasets(deepfake_types):
    deepfake_datasets = {}
    for deepfake_type in deepfake_types:
        train_loader, val_loader, test_loader = load_dataset(deepfake_type)
        deepfake_datasets[deepfake_type] = {'train': train_loader, 'val': val_loader, 'test': test_loader, 'train_iter': iter(train_loader)}
    return deepfake_datasets




# clients
def create_clients(deepfake_datasets, initial='clients'): 
    #create a list of client names
    client_names = ['{}_{}'.format(initial, deepfake_type) for deepfake_type in deepfake_types]
    #shard data and place at each client. each shard is a different deepfake type
    shards = deepfake_datasets

    #number of clients must equal number of shards
    assert(len(shards) == len(client_names))

    return {'{}_{}'.format(initial, deepfake_type) : shards[deepfake_type] for deepfake_type in deepfake_types}

def get_client_datasizes(deepfake_types, initial='client'):
    return {'{}_{}'.format(initial, deepfake_type) : get_dataset_size(deepfake_type) for deepfake_type in deepfake_types}



# test model
def test_model(model, test_loader, loss_fn):
    model.eval()  # Set model to evaluation mode
    total_loss = 0.0
    correct = 0
    total = 0
    criterion = loss_fn

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Model forward pass
            outputs = model(inputs)
            
            # Ensure outputs and labels have the same shape
            outputs = outputs.view(-1)  # Flatten outputs to match labels
            labels = labels.view(-1).float()  # Flatten labels and convert to float

            # Compute loss
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            # Calculate predictions and accuracy
            predictions = (torch.sigmoid(outputs) > 0.5).long()
            correct += (predictions == labels.long()).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    print('test acc: {:.3%}, loss: {}'.format(accuracy, avg_loss))
    return avg_loss, accuracy


def load_full_dataset():
    transformer = ImageTransform(IMG_SIZE, mean, std)
    train_datasets = []
    val_datasets = []
    test_datasets = []
    for deepfake_type in deepfake_types:
        TRAIN_FOLDER = fr'D:\FF\crops\{deepfake_type}\train' 
        VAL_FOLDER = fr'D:\FF\crops\{deepfake_type}\val' 
        TEST_FOLDER = fr'D:\FF\crops\{deepfake_type}\test' 

        train_ds = ImageFolder(root=TRAIN_FOLDER, transform=transformer)
        val_ds = ImageFolder(root=VAL_FOLDER, transform=transformer)
        test_ds = ImageFolder(root=TEST_FOLDER, transform=transformer)

        train_datasets.append(train_ds)
        val_datasets.append(val_ds)
        test_datasets.append(test_ds)
    full_train_ds = ConcatDataset(train_datasets)
    full_val_ds = ConcatDataset(val_datasets)
    full_test_ds = ConcatDataset(test_datasets)
    train_loader = DataLoader(full_train_ds, shuffle=True, batch_size=64, num_workers=8)
    val_loader = DataLoader(full_val_ds, shuffle=True, batch_size=64, num_workers=8)
    test_loader = DataLoader(full_test_ds, shuffle=False, batch_size=64, num_workers=8)
    return train_loader, val_loader, test_loader


PATIENCE = 5
MODEL_NAME = f'ff_test_fl_effnet_fedsgd_5000comm'

def save_checkpoint(model, epoch, loss_history, acc_history, filename=MODEL_NAME):
    checkpoint = {
        "model_state": model.state_dict(),
        "epoch": epoch,
        "loss_history": loss_history,
        "acc_history": acc_history,
    }
    print('hellooo')
    filepath = f"{filename}.pth"
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved to {filepath}")

class EarlyStopper:
    def __init__(self, patience=PATIENCE, verbose=True):
        self.patience = patience
        self.counter = 0
        self.best_score = float('inf')  # Initialize to positive infinity
        self.verbose = verbose
    def best_val(self, val_loss):
        if val_loss < self.best_score:
            self.best_score = val_loss
            self.counter = 0
            return True
        else:
            self.counter += 1
            return False
    def early_stop(self):
        if self.counter >= self.patience:
            if self.verbose:
                print("Early stopping...")
            return True
        return False
from torch.optim import SGD
# train
def train_model_on_shard(minibatches, model, train_iter, train_loader, val_loader):
    torch.cuda.empty_cache()
    local_optimizer = SGD(model.parameters(), lr=lr, momentum=0.9)
    gradients = None
    num_batches = 0  # Track number of batches for averaging
    # test_model(model, val_loader, loss_fn)

    for minibatch in range(minibatches): 
        data, labels = get_next_batch(train_iter, train_loader)
        print('', labels)
        running_loss = 0
        correct = 0
        total = 0
        model.train()
        data = data.squeeze(0)
        labels = labels.squeeze(0)
        labels = labels.to(device).float()
        data = data.to(device) 
        local_optimizer.zero_grad()  # clear previous gradients

        #data = data.to(device).float()
        outputs = model(data).squeeze()
        losses = loss_fn(outputs, labels)
        running_loss += losses.item()
        
        losses.backward()

        # local_optimizer.step()
        # print([torch.zeros_like(param.grad) for param in model._fc.parameters()])
        if gradients is None:
            gradients = [torch.zeros_like(param.grad) for param in model._fc.parameters()]
        for i, param in enumerate(model._fc.parameters()):
            gradients[i] += param.grad.clone()

        predicted   = (torch.sigmoid(outputs) >= 0.5).float()
        correct += (predicted == labels).sum().item() 
        total += labels.size(0)

        num_batches += 1
        average_train_loss = running_loss / total
        average_train_accuracy = correct / total
        print('train', average_train_accuracy, average_train_loss)
    #gradients = [g / num_batches for g in gradients]
    # Manually update the gradients with averaged gradients
    #for param, grad in zip(model._fc.parameters(), gradients):
    #    param.grad = grad
        # local_optimizer.step()
        # test_model(model, val_loader, loss_fn)
    
    return gradients

def federated_sgd(gradients_list):
    avg_gradients = []

    num_clients = len(gradients_list)

    for i in range(len(gradients_list[0])):
        avg_grad = sum([gradients[i] for gradients in gradients_list]) / num_clients
        avg_gradients.append(avg_grad)

    return avg_gradients



def get_next_batch(train_iter, train_loader):
    try:
        return next(train_iter)  # Fetch the next batch
    except StopIteration: 
        # If iterator is exhausted, recreate it and get the first batch
        train_iter = iter(train_loader)
        return next(train_iter)
    
import copy
if __name__ == '__main__':
    device = 'cuda'
    comms_round = 1000
    lr = 0.01
    loss=torch.nn.BCEWithLogitsLoss()

    global_model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=1)
    for param in global_model.parameters():
        param.requires_grad = False
    # Unfreeze the final fully connected layer
    for param in global_model._fc.parameters():
        param.requires_grad = True
    global_model = global_model.to(device)
    global_model.train()
    print([p for p in global_model._fc.parameters()])
    optimizer = SGD(global_model.parameters(), lr=0.01, momentum=0.9)

    train_batched, _, test_batched = load_full_dataset()
    train_batched_size = len(train_batched)

    


    deepfake_datasets = load_datasets(deepfake_types)
    clients = create_clients(deepfake_datasets, initial='client')

    global_loss_histories = []
    global_acc_histories = []
    for comm_round in range(comms_round):  
        all_gradients = []
        print(comm_round, 'comm')
        clients_batched = clients
        client_names= list(clients_batched.keys())

        for client in client_names:
            print(client)
            local_model = cloned_model = copy.deepcopy(global_model)
            local_model = local_model.to(device) 
            
            train_loader, val_loader, test_loader, train_iter = clients_batched[client]['train'], clients_batched[client]['val'], clients_batched[client]['test'], clients_batched[client]['train_iter']
            model_gradients = train_model_on_shard(1, local_model, train_iter, train_loader, val_loader)
            all_gradients.append(model_gradients)


        avg_gradients = federated_sgd(all_gradients)
        print('average+gradients', avg_gradients)

        # update global model
        with torch.no_grad(): 
            print('update global')
            for param, avg_grad in zip(global_model._fc.parameters(), avg_gradients):
                param.grad = avg_grad.detach()
            optimizer.step()
            optimizer.zero_grad()


        #test global model and print out metrics after each communications round
        print('global')
        # print([p for p in global_model._fc.parameters()])
        # global_loss, global_acc = test_model(global_model, train_loader, loss_fn)
        #global_loss_histories.append(global_loss)
        #global_acc_histories.append(global_acc)

        print('save checkpoint')
        save_checkpoint(global_model, comm_round, global_loss_histories, global_acc_histories)