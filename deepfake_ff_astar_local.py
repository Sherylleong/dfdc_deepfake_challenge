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


DEEPFAKE_TYPE = 'FaceSwap'
SOURCE_FOLDER= r"D:\FF"
TRAIN_FOLDER = fr'D:\FF\crops\{DEEPFAKE_TYPE}\train' 
VAL_FOLDER = fr'D:\FF\crops\{DEEPFAKE_TYPE}\val' 
TEST_FOLDER = fr'D:\FF\crops\{DEEPFAKE_TYPE}\test' 

SAVE_PATH = fr'models\{DEEPFAKE_TYPE}'

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
MODEL_NAME = f'ff_effnet0_1fc_{DEEPFAKE_TYPE}'

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
        if val_loss >= self.best_score:
            self.counter += 1
            return False
        return True
    def early_stop(self):
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

        if earlystopper.best_val(average_val_loss):
            histories = {'train_acc': train_accuracy_history, 'val_acc': val_accuracy_history, 'train_loss': train_loss_history, 'val_loss':  val_loss_history}
            save_checkpoint(model, optimizer, epoch, histories)
        if earlystopper.early_stop():
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
    train_ds = ImageFolder(root=TRAIN_FOLDER, transform=transformer)
    val_ds = ImageFolder(root=VAL_FOLDER, transform=transformer)
    test_ds = ImageFolder(root=TEST_FOLDER, transform=transformer)

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