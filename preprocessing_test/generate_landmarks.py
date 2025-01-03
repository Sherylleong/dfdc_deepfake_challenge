import argparse
import os
from functools import partial
from multiprocessing.pool import Pool



os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

from tqdm import tqdm


import cv2
from os_helper import *
cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)
from os_helper import *

from PIL import Image
from facenet_pytorch.models.mtcnn import MTCNN
import numpy as np

detector = MTCNN(margin=0, thresholds=[0.65, 0.75, 0.75], device="cpu")

images_dir = 'dfdc_train_part_0'
def save_landmarks(ori_id, root_dir):
    ori_id = ori_id[:-4]  # Remove the last four characters from ori_id, typically a file extension
    ori_dir = os.path.join(root_dir, "crops", ori_id) # Directory containing cropped images
    landmark_dir = os.path.join(root_dir, "landmarks", ori_id) # Directory to save landmarks
    os.makedirs(landmark_dir, exist_ok=True)
    for frame in range(320): # Iterate over the first 320 frames
        if frame % 10 != 0: # Process every 10th frame
            continue
        for actor in range(2): # Assuming there are 2 actors
            image_id = "{}_{}.png".format(frame, actor) # Create the image filename
            landmarks_id = "{}_{}".format(frame, actor) # Create the landmark filename
            ori_path = os.path.join(ori_dir, image_id) # Full path to the image
            landmark_path = os.path.join(landmark_dir, landmarks_id) # Full path to save landmarks
            if os.path.exists(ori_path): # Check if the original image exists
                try:
                    image_ori = cv2.imread(ori_path, cv2.IMREAD_COLOR)[...,::-1] # Read the image
                    frame_img = Image.fromarray(image_ori)
                    batch_boxes, conf, landmarks = detector.detect(frame_img, landmarks=True) # Detect landmarks
                    if landmarks is not None: # Check if landmarks were detected
                        landmarks = np.around(landmarks[0].astype(float)).astype(np.int16) # Save landmarks as a .npy file
                        np.save(landmark_path, landmarks)
                except Exception as e:
                    print(e)
                    pass

images_dir = 'dfdc_train_part_0'
root_dir = os.getcwd()
def main():
    ids = get_videos_basenames_from_folder(os.path.join(root_dir, images_dir))
    os.makedirs(os.path.join(root_dir, "landmarks"), exist_ok=True)
    with Pool(processes=4) as p:
        with tqdm(total=len(ids)) as pbar:
            func = partial(save_landmarks, root_dir=root_dir)
            for v in p.imap_unordered(func, ids):
                pbar.update()


if __name__ == '__main__':
    main()
