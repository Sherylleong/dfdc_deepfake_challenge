import os
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import matplotlib.pyplot as plt
from collections import OrderedDict
from typing import List
from os import cpu_count
import cv2
cv2.ocl.setUseOpenCL(False) # use cpu instead of gpu
cv2.setNumThreads(0)
from tqdm import tqdm
from PIL import Image
from facenet_pytorch.models.mtcnn import MTCNN
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import json
from functools import partial
from glob import glob
from multiprocessing.pool import Pool
from os_helper import *

class FacenetDetector:
    def __init__(self, batch_size=32, device="cuda:0") -> None:
        self.batch_size = batch_size
        self.detector = MTCNN(margin=0,thresholds=[0.85, 0.95, 0.95], device=device)

    def detect_faces(self, frames) -> List:
        batch_boxes, *_ = self.detector.detect(frames, landmarks=False)
        return [b.tolist() if b is not None else None for b in batch_boxes]




class VideoDataset(Dataset):
    def __init__(self, videos) -> None:
        super().__init__()
        self.videos = videos

    def __getitem__(self, index: int):
        video = self.videos[index]
        capture = cv2.VideoCapture(video)
        frames_num = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) # this one just gets all the frames
        frames = OrderedDict()
        for i in range(frames_num):
            capture.grab()
            success, frame = capture.retrieve()
            if not success:
                continue
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
            frame = frame.resize(size=[s // 2 for s in frame.size]) # resize to half the size
            frames[i] = frame
        return video, list(frames.keys()), list(frames.values())

    def __len__(self) -> int:
        return len(self.videos)


def extract_video_frames(video):
    capture = cv2.VideoCapture(video)
    frames_num = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for i in range(frames_num):
        capture.grab()
        success, frame = capture.retrieve()
        frame.append(frame)
        if not success:
            continue
    return frames


def custom_collate_fn(batch):
    return batch

root_dir = os.getcwd()
def process_videos(videos, detector=FacenetDetector()):
    dataset = VideoDataset(videos)
    loader = DataLoader(dataset, shuffle=False, batch_size=1, collate_fn=custom_collate_fn) # process video 1 by 1
    for item in tqdm(loader):
        result = {}
        video, indices, frames = item[0]
        batches = [frames[i:i + detector.batch_size] for i in range(0, len(frames), detector.batch_size)]
        for j, frames in enumerate(batches):
            result.update({int(j * detector.batch_size) + i : b for i, b in zip(indices, detector.detect_faces(frames))})
        id = os.path.splitext(os.path.basename(video))[0] # remove basename and keep filename
        out_dir = os.path.join(root_dir, "boxes")
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "{}.json".format(id)), "w") as f:
            json.dump(result, f) # raw bounding box data for each video for frame


def main():
    video_files = get_videos_from_folder(os.path.join(os.getcwd(), 'dfdc_train_part_0'))
    process_videos(video_files)


if __name__ == "__main__":
    main()
