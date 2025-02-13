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
import torch
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

### LOAD DATA
deepfake_types = ['Deepfakes', 'Face2Face', 'FaceShifter', 'FaceSwap', 'NeuralTextures']
IMG_SIZE = 224
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
BATCH_SIZE = 64
device = 'cuda'
EPOCHS = 100
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
    
def load_dataset(DEEPFAKE_TYPE):
    transformer = ImageTransform(IMG_SIZE, mean, std)

    TRAIN_FOLDER = fr'D:\FF\crops\{DEEPFAKE_TYPE}\train' 
    VAL_FOLDER = fr'D:\FF\crops\{DEEPFAKE_TYPE}\val' 
    TEST_FOLDER = fr'D:\FF\crops\{DEEPFAKE_TYPE}\test' 

    train_ds = ImageFolder(root=TRAIN_FOLDER, transform=transformer)
    val_ds = ImageFolder(root=VAL_FOLDER, transform=transformer)
    test_ds = ImageFolder(root=TEST_FOLDER, transform=transformer)

    train_loader = DataLoader(train_ds, shuffle=True, batch_size=64, num_workers=8)
    val_loader = DataLoader(val_ds, shuffle=True, batch_size=64, num_workers=8)
    test_loader = DataLoader(test_ds, shuffle=False, batch_size=64, num_workers=8)
    return train_loader, val_loader, test_loader

def load_datasets(deepfake_types):
    deepfake_datasets = {}
    for deepfake_type in deepfake_types:
        train_loader, val_loader, test_loader = load_dataset(deepfake_type)
        deepfake_datasets[deepfake_type] = {'train': train_loader, 'val': val_loader, 'test': test_loader}
    return deepfake_datasets

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

deepfake_datasets = load_datasets(deepfake_types)

### CREATE CLIENTS
def create_clients(deepfake_datasets, initial='clients'):
    #create a list of client names
    client_names = ['{}_{}'.format(initial, deepfake_type) for deepfake_type in deepfake_types]
    #shard data and place at each client. each shard is a different deepfake type
    shards = deepfake_datasets

    #number of clients must equal number of shards
    assert(len(shards) == len(client_names))

    return {'{}_{}'.format(initial, deepfake_type) : shards[deepfake_type] for deepfake_type in deepfake_types}

clients = create_clients(deepfake_datasets, initial='client')
clients_batched = clients

### MODEL

class SimpleMLP:
    @staticmethod
    def build(shape, classes,only_digits=True):
        model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=1)
        return model

def weight_scaling_factor(clients_trn_data, client_name):
    client_names = list(clients_trn_data.keys())
    #get the bs
    bs = clients_trn_data[client_name].batch_size
    #first calculate the total training data points across clients
    global_count = sum([len(clients_trn_data[client_name]) for client_name in client_names])*bs
    # get the total number of data points held by a client
    local_count =  len(clients_trn_data[client_name]) * bs
    return local_count/global_count


def scale_model_weights(weight_dict, scalar):
    '''function for scaling a models weights'''
    for key in weight_dict:
        weight_dict[key] *= scalar  # Scale weights in-place
    return weight_dict

def scale_model_weights2(weight, scalar):
    '''function for scaling a models weights'''
    weight_final = []
    steps = len(weight)
    for i in range(steps):
        weight_final.append(scalar * weight[i])
    return weight_final


def sum_scaled_weights(scaled_weight_list):
    '''Return the sum of the listed scaled weights. The is equivalent to scaled avg of the weights'''
    avg_grad = list()
    #get the average grad accross all client gradients
    for grad_list_tuple in zip(*scaled_weight_list):
        layer_mean = torch.sum(torch.stack(grad_list_tuple), dim=0)
        avg_grad.append(layer_mean)
        
    return avg_grad


def test_model_mid(model, test_loader, loss_fn):
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

    avg_loss = total_loss / len(test_loader)
    accuracy = correct / total
    print('acc: {:.3%} | loss: {}'.format(accuracy, avg_loss))
    return avg_loss, accuracy

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

    avg_loss = total_loss / len(test_loader)
    accuracy = correct / total
    print('acc: {:.3%} | loss: {}'.format(accuracy, avg_loss))
    return avg_loss, accuracy

### TRAIN
PATIENCE = 5
MODEL_NAME = f'ff_effnet0_1fc_fl'

def save_checkpoint(model, epoch, local_loss_history, local_acc_history, global_loss_history, global_acc_history, filename=MODEL_NAME):
    checkpoint = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "epoch": epoch,
        "local_loss_history": local_loss_history,
        "local_acc_history": local_acc_history,
        "global_loss_history": global_loss_history,
        "global_acc_history": global_acc_history
    }
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

def train(epochs, optimizer, model, train_loader, val_loader):
    torch.cuda.empty_cache()
    train_loss_history = []
    train_accuracy_history = []
    val_loss_history = []
    val_accuracy_history = []
    for epoch in range(epochs): 
        running_loss = 0
        correct = 0
        total = 0
        model.train()
        for data, labels in train_loader:
            data = data.squeeze(0)
            labels = labels.squeeze(0)
            labels = labels.to(device).float()
            data = data.to(device)
            optimizer.zero_grad()  # clear previous gradients

            #data = data.to(device).float()
            outputs = model(data).squeeze()
            losses = loss_fn(outputs, labels)
            running_loss += losses.item()  # accumulate loss
            
            losses.backward()
            optimizer.step()
            predicted   = (torch.sigmoid(outputs) >= 0.5).float() # calculate if label is 0 or 1
            correct += (predicted == labels).sum().item() 
            total += labels.size(0)
            train_loss_history.append(losses.item())

        average_train_loss = running_loss / total
        average_train_accuracy = correct / total
        train_loss_history.append(average_train_loss)
        train_accuracy_history.append(average_train_accuracy)

        model.eval()
        running_loss = 0
        correct = 0
        total = 0 
        for data, labels in val_loader:
            data = data.squeeze(0)
            labels = labels.squeeze(0)
            labels = labels.to(device).float()
            data = data.to(device).float()

            outputs = model(data).squeeze()
            losses = loss_fn(outputs, labels)
            running_loss += losses.item()  # accumulate loss
            predicted = (torch.sigmoid(outputs) >= 0.5).float() # calculate if label is 0 or 1
            correct += (predicted == labels).sum().item() 
            total += labels.size(0)
        average_val_loss = running_loss / total
        average_val_accuracy = correct / total
        val_loss_history.append(average_val_loss)
        val_accuracy_history.append(average_val_accuracy)
        print(f"Epoch [{epoch}/{epochs}], Train Loss: {average_train_loss:.5f}, Train Accuracy: {average_train_accuracy:.5f}, Val Loss: {average_val_loss:.5f}, Val Accuracy: {average_val_accuracy:.5f}")

        
    return train_loss_history, val_loss_history, train_accuracy_history, val_accuracy_history











if __name__ == '__main__':
    #create optimizer
    import copy
    comms_round = 10
    lr = 0.01
    loss=torch.nn.BCEWithLogitsLoss()
    global_loss_list = []
    global_acc_list = []
    best_test_acc = 0

    global_model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=1)
    for param in global_model.parameters():
        param.requires_grad = False
    # Unfreeze the final fully connected layer
    for param in global_model._fc.parameters():
        param.requires_grad = True
    global_model = global_model.to(device)

    _, val_batched, test_batched = load_full_dataset()

    average_weights=copy.deepcopy(global_model.state_dict())

    local_acc_comp = []
    local_loss_comp = []

    for comm_round in range(comms_round):  
        print(comm_round, 'comm')
        scaled_local_weight_list = list()
        client_names= list(clients_batched.keys())
        local_accs = []
        local_losses = []
        average_weights = {key: torch.zeros_like(param) for key, param in global_model.state_dict().items()}
        for client in client_names:
            print(client)
            local_model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=1)
            for param in local_model.parameters():
                param.requires_grad = False
            # Unfreeze the final fully connected layer
            for param in local_model._fc.parameters():
                param.requires_grad = True
            global_weights = copy.deepcopy(global_model.state_dict())
            local_model.load_state_dict(global_weights)
            optimizer = torch.optim.Adam(local_model.parameters(), lr=LR)
            local_model = local_model.to(device)
            train_loader, val_loader, test_loader = clients_batched[client]['train'], clients_batched[client]['val'], clients_batched[client]['test']
            train_loss_history, val_loss_history, train_accuracy_history, val_accuracy_history = train(2, optimizer, local_model, train_loader, val_loader)
            weights=local_model.state_dict()
            for key in weights:
                average_weights[key] += weights[key]
            
            #local_loss, local_acc = test_model(local_model, test_batched, loss_fn)
            #local_accs.append(local_acc)
            #local_losses.append(local_loss)

        for key in average_weights:
            average_weights[key] = average_weights[key] / 5

        #clear session to free memory after each communication round
        torch.cuda.empty_cache()

        #local_acc_comp.append(np.mean(local_acc))
        #local_loss_comp.append(np.mean(local_loss))


        #update global model
        global_model.load_state_dict(average_weights)

        #test global model and print out metrics after each communications round
        print('global')
        global_acc, global_loss = test_model(global_model, val_batched, loss_fn)
        global_loss_list.append(global_loss)
        global_acc_list.append(global_acc)
        if global_acc > best_test_acc:
            best_test_acc = global_acc
            save_checkpoint(global_model, comm_round, local_loss_comp, local_acc_comp, global_loss_list, global_acc_list, filename=MODEL_NAME)
    