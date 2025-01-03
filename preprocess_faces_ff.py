import os
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
# Suppress all warnings

# from tqdm import tqdm_notebook
# from google.colab.patches import cv2_imshow
from IPython.display import HTML #imports to play videos
from base64 import b64encode 
#from skimage.measure import compare_ssim
import glob
import time
from PIL import Image
from tqdm import tqdm
from functools import partial
from PIL import Image
from multiprocessing import Pool
import cv2
import skimage.measure
#import albumentations as A
from tqdm.notebook import tqdm 
#from albumentations.pytorch import ToTensor

from facenet_pytorch import MTCNN



import cv2
import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
from torchvision import models, transforms

import matplotlib.pyplot as plt

plt.rc('font', size=14)
plt.rc('axes', labelsize=14, titlesize=14)
plt.rc('legend', fontsize=14)
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)


device = 'cuda'


SOURCE_FOLDER= r"D:"

ORI_FOLDER = r'original_sequences\youtube\c23\videos'
MANIP_FOLDER = r'manipulated_sequences\Deepfakes\c23\videos'
ORI_BOXES_FOLDER = r'original_sequences'
MANIP_BOXES_FOLDER = r'manipulated_sequences'


IMG_SIZE = 224
BATCH_SIZE = 64
EPOCHS = 10


frames_per_video = 32

img_size = 224
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
class ImageTransform:
    def __init__(self, size, mean, std):
        self.data_transform = transforms.Compose([
                transforms.Resize((size, size), interpolation=Image.BILINEAR),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])

    def __call__(self, img):
        return self.data_transform(img)

import json
def extract_bboxes_from_video(path):
    bboxes = json.load(path) # load bboxes
def extract_frames_from_video(path, n_frames=32):
    print('extracting frames', path)
    # Create video reader and find length
    v_cap = cv2.VideoCapture(path)
    v_len = int(v_cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Pick 'n_frames' evenly spaced frames to sample
    if n_frames is None:
        sample = np.arange(0, v_len)
    else:
        sample = np.linspace(0, v_len - 1, n_frames).astype(int)
    # Loop through frames
    frames = []
    for j in range(v_len):
        success = v_cap.grab()
        if j in sample:
            # Load frame
            success, frame = v_cap.retrieve()
            if not success:
                continue
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
    return frames


def get_faces_from_video_image(video_path, bbox_path):
    transformer = ImageTransform(img_size, mean, std)
    faces = []
    # Movie to Image
    try:
        frames = extract_frames_from_video(video_path)
    except:
        frames = []
    if len(frames) == 0:
        return []
    # get bboxes
    try:
        with open(bbox_path, 'r') as file:
            bboxes = json.load(file)
    except Exception as e:
        print(e)
        bboxes = []
    if len(bboxes) == 0:
        return []
    
    # crop Faces
    print('crop faces', video_path)
    _frame = np.array(frames)
    for i in range(len(frames)):
        frame = frames[i]
        if frame is None:
            continue
        boxes = bboxes
        try: 
            x = int(boxes[i][0])
            y = int(boxes[i][1])
            z = int(boxes[i][2])
            w = int(boxes[i][3])
            face = frame[y:w, x:z]
            
            # Preprocessing
            face = Image.fromarray(face)
            # face = transformer(face)
            
            faces.append(face)

        except Exception as e:
            print(1111,e)
            pass
            #faces.append(None)
        
    # Padding None
    #faces = [c for c in faces if c is not None]
    
    return faces

def get_faces_from_video(video_path, bbox_path):
    transformer = ImageTransform(img_size, mean, std)
    faces = []
    # Movie to Image
    try:
        frames = extract_frames_from_video(video_path)
    except:
        frames = []
    if len(frames) == 0:
        return []
    # get bboxes
    try:
        with open(bbox_path, 'r') as file:
            bboxes = json.load(file)
    except Exception as e:
        print(e)
        bboxes = []
    if len(bboxes) == 0:
        return []
    
    # crop Faces
    print('crop faces', video_path)
    _frame = np.array(frames)
    for i in range(len(frames)):
        frame = frames[i]
        if frame is None:
            continue
        boxes = bboxes
        try: 
            x = int(boxes[i][0])
            y = int(boxes[i][1])
            z = int(boxes[i][2])
            w = int(boxes[i][3])
            face = frame[y:w, x:z]
            
            # Preprocessing
            face = Image.fromarray(face)
            #face = transformer(face)
            
            faces.append(face)

        except Exception as e:
            print(e)
            pass
            #faces.append(None)
        
    # Padding None
    #faces = [c for c in faces if c is not None]
    
    return faces
import torch
def save_face_crops(faces, video_save_folder):
    
    for i in range(len(faces)):
        face = faces[i]
        #face = Image.fromarray(face.numpy())
        #print(face.shape)
        # Save the image
        face_save_path = video_save_folder + f'/tensor_{i}.pt'
        torch.save(face, face_save_path)
        
        #face.save(os.path.join(video_save_folder, "{}.png".format(i)))
        # cv2.imwrite(os.path.join(video_save_folder, "{}.png".format(i)), faces[i].numpy())

def save_face_crops_image(faces, save_folder, video_name):
    for i in range(len(faces)):
        face = faces[i]
        #face = Image.fromarray(face.numpy())
        #print(face.shape)
        # Save the image
        print(os.path.join(save_folder, f"{video_name}_{i}.png"))
        face.save(os.path.join(save_folder, f"{video_name}_{i}.png")) 

def get_video_names_from_folder(folder_path):
    # List all files in the specified folder
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    if 'metadata.json' in files:
        files.remove('metadata.json')
    return files

import concurrent.futures


def preprocess_image(source_folder, to_folder, boxes_folder, video_name):
    print(f"Started processing: {video_name}")
    try:
        video_path = os.path.join(SOURCE_FOLDER, source_folder, video_name)
        video_save_path = os.path.join(SOURCE_FOLDER, to_folder)
        bbox_path = os.path.join(SOURCE_FOLDER, 'extracted_boxes', boxes_folder,  video_name[:-4] + '.json')
        print('creating dir, ',video_name)
        os.makedirs(video_save_path, exist_ok=True)
        print('getting faces, ',video_name)
        faces = get_faces_from_video(video_path, bbox_path)
        print('savning faces, ',video_name)
        save_face_crops_image(faces, video_save_path, video_name)
    except Exception as e:
        print(e)
    print(f"Finished processing: {video_name}")



SOURCE_FOLDER= r"D:\FF"


ORI_VIDEOS_FOLDER = r'original_sequences\youtube\c23\videos'
MANIP_VIDEOS_FOLDER = r'manipulated_sequences\Deepfakes\c23\videos'
ORI_FACES_FOLDER = r'original_sequences'
MANIP_FACES_FOLDER = r'manipulated_sequences'

if __name__ == '__main__':

    source_folder = ORI_VIDEOS_FOLDER
    to_folder = ORI_FACES_FOLDER
    boxes_folder = os.path.join(SOURCE_FOLDER, 'extracted_boxes', ORI_BOXES_FOLDER)

    out_dir = os.path.join(SOURCE_FOLDER, 'crops', to_folder)

    os.makedirs(out_dir, exist_ok=True)
    video_names = get_video_names_from_folder(os.path.join(SOURCE_FOLDER, source_folder))
    #for video_name in video_names:
    #    preprocess_image(source_folder, out_dir, video_name)
    with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
        list(tqdm(executor.map(partial(preprocess_image, source_folder, out_dir, boxes_folder), video_names)))


