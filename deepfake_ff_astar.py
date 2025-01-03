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

from PIL import Image
#from albumentations.pytorch import ToTensor 
from torch.utils.data import Dataset, random_split
from torch.utils.data import DataLoader
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


DATA_FOLDER_DFDC = r'/raid/soumik/deepFake/dataset/dfdc/images'
DATA_FOLDER_FF = r'/raid/soumik/deepFake/dataset/ff_working/ff_dataset_23'
DATA_FOLDER_FF = r'/raid/sheryl/ff_dataset_23'
#DATA_FOLDER_FF = r'D:\FF\testastar'
SAVE_PATH = r'/raid/sheryl/models'
#SAVE_PATH = r'D:\FF'


IMG_SIZE = 224
BATCH_SIZE = 64
EPOCHS = 100

mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)


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
MODEL_NAME = 'best_model_ff_effnet0_1fc.pth'


def save_checkpoint(model, optimizer, save_path, epoch):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
    }, save_path)

class EarlyStopper:
    def __init__(self, patience=PATIENCE, verbose=True):
        self.patience = patience
        self.counter = 0
        self.best_score = float('inf')  # Initialize to positive infinity
        self.verbose = verbose

    def early_stop(self, val_loss, model, optimizer, epoch):
        if val_loss < self.best_score:
            self.best_score = val_loss  # Update best validation loss
            self.counter = 0  # Reset patience counter
            save_checkpoint(model, optimizer, os.path.join(SAVE_PATH, MODEL_NAME), epoch)  # Save the best model
        else:
            self.counter += 1  # Increment patience counter
           
        if self.counter >= self.patience:
            if self.verbose:
                print("Early stopping...")
            return True
        return False
    

'''
TRAINING
'''
EPOCHS = 1000
LR = 1e-5

def train(epochs, optimizer, model, train_loader, val_loader, earlystopper):
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


        if earlystopper.early_stop(average_val_loss, model, optimizer, epoch):
            break 
        print(f"Epoch [{epoch}/{epochs}], Train Loss: {average_train_loss:.5f}, Train Accuracy: {average_train_accuracy:.5f}, Val Loss: {average_val_loss:.5f}, Val Accuracy: {average_val_accuracy:.5f}")
    return train_loss_history, val_loss_history, train_accuracy_history, val_accuracy_history



if __name__ == '__main__':
    device = 'cuda:0' 
    
    model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=1)
    model = model.to(device)

    transformer = ImageTransform(IMG_SIZE, mean, std)
    earlystopper = EarlyStopper()
    # Initialize dataset and dataloader
    torch.manual_seed(0)
    data_dataset = ImageFolder(root=DATA_FOLDER_FF, transform=transformer)
    n = len(data_dataset)  # total number of examples
    n_val = int(np.floor(0.2 * n))  # take ~20% for val
    n_test = int(np.floor(0.1 * n))  # take ~10% for test
    n_train = n - n_val - n_test

    train_ds, val_ds, test_ds = random_split(data_dataset, [n_train, n_val, n_test])

    train_loader = DataLoader(train_ds, shuffle=True, batch_size=64, num_workers=4)
    val_loader = DataLoader(val_ds, shuffle=True, batch_size=64, num_workers=4)
    test_loader = DataLoader(test_ds, shuffle=False, batch_size=64, num_workers=4)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.BCEWithLogitsLoss()
    train_loss_history, val_loss_history, train_accuracy_history, val_accuracy_history = train(EPOCHS, optimizer, model, train_loader, val_loader, earlystopper)
    np.save(os.path.join(SAVE_PATH, 'train_loss_history.npy'), np.asarray(train_loss_history))
    np.save(os.path.join(SAVE_PATH, 'val_loss_history.npy'), np.asarray(val_loss_history))
    np.save(os.path.join(SAVE_PATH, 'train_accuracy_history.npy'), np.asarray(train_accuracy_history))
    np.save(os.path.join(SAVE_PATH, 'val_accuracy_history.npy'), np.asarray(val_accuracy_history))