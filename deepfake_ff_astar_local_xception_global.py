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
import timm
from torchvision import models
from PIL import Image
#from albumentations.pytorch import ToTensor 
from torch.utils.data import Dataset, random_split
from torch.utils.data import DataLoader, ConcatDataset
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import models
import sys
import sklearn

import cv2
import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
from torchvision import models, transforms
from torch.utils.data import WeightedRandomSampler
 
import matplotlib.pyplot as plt 



SOURCE_FOLDER= r"D:\FF"
deepfake_types = ['Deepfakes', 'Face2Face', 'FaceShifter', 'FaceSwap', 'NeuralTextures']


IMG_SIZE = 299
BATCH_SIZE = 64
EPOCHS = 10

mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)

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
    train_loader = DataLoader(full_train_ds, shuffle=True, batch_size=64, num_workers=4)
    val_loader = DataLoader(full_val_ds, shuffle=True, batch_size=64, num_workers=4)
    test_loader = DataLoader(full_test_ds, shuffle=False, batch_size=64, num_workers=4)
    return train_loader, val_loader

    
    
'''
IMAGE TRANSFORMS
'''
class ImageTransform:
    def __init__(self, size, mean, std):
        self.data_transform = transforms.Compose([
                transforms.Resize((size, size), interpolation=Image.BILINEAR),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])

    def __call__(self, img):
        return self.data_transform(img)



'''
EARLY STOPPER
'''
PATIENCE = 5
MODEL_NAME = f'ff_xceptionnet_global_10epoch'

def save_checkpoint(model, optimizer, epoch, history, filename=MODEL_NAME):
    checkpoint = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "epoch": epoch,
        "history": history,
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

'''
TRAINING
'''
LR = 0.001

from torch.amp import autocast, GradScaler
def train(epochs, optimizer, model, train_loader, val_loader, earlystopper):
    print('train start')
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
        print('train end')

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
        print('val end')
        average_val_loss = running_loss / total
        average_val_accuracy = correct / total
        val_loss_history.append(average_val_loss)
        val_accuracy_history.append(average_val_accuracy)
        print(f"Epoch [{epoch}/{epochs}], Train Loss: {average_train_loss:.5f}, Train Accuracy: {average_train_accuracy:.5f}, Val Loss: {average_val_loss:.5f}, Val Accuracy: {average_val_accuracy:.5f}")
        if earlystopper.best_val(average_val_loss):
            print('new best')
            histories = {'train_acc': train_accuracy_history, 'val_acc': val_accuracy_history, 'train_loss': train_loss_history, 'val_loss':  val_loss_history}
            save_checkpoint(model, optimizer, epoch, histories)
        print(earlystopper.counter, earlystopper.patience)
        if earlystopper.early_stop():
            print('STOP')
            break 
        
    return train_loss_history, val_loss_history, train_accuracy_history, val_accuracy_history

def trainepoch(epochs, optimizer, model, train_loader, val_loader, earlystopper):
    print('train start')
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
        print('train end')

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
        print('val end')
        average_val_loss = running_loss / total
        average_val_accuracy = correct / total
        val_loss_history.append(average_val_loss)
        val_accuracy_history.append(average_val_accuracy)
        print(f"Epoch [{epoch}/{epochs}], Train Loss: {average_train_loss:.5f}, Train Accuracy: {average_train_accuracy:.5f}, Val Loss: {average_val_loss:.5f}, Val Accuracy: {average_val_accuracy:.5f}")
        histories = {'train_acc': train_accuracy_history, 'val_acc': val_accuracy_history, 'train_loss': train_loss_history, 'val_loss':  val_loss_history}
        save_checkpoint(model, optimizer, epoch, histories)
        
    return train_loss_history, val_loss_history, train_accuracy_history, val_accuracy_history

if __name__ == '__main__':
    device = 'cuda' 
    model = timm.create_model('xception', pretrained=True, num_classes=1)
    # Freeze all layers
    for param in model.parameters():
        param.requires_grad = False
    # Unfreeze the final fully connected layer
    for param in model.fc.parameters():
        param.requires_grad = True
    model = model.to(device)


    transformer = ImageTransform(IMG_SIZE, mean, std)
    earlystopper = EarlyStopper()
    # Initialize dataset and dataloader
    torch.manual_seed(0)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    train_loader, val_loader = load_full_dataset()
    train_loss_history, val_loss_history, train_accuracy_history, val_accuracy_history = trainepoch(EPOCHS, optimizer, model, train_loader, val_loader, earlystopper)

