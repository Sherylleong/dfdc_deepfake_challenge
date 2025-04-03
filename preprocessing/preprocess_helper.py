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

def crop_image(frame, bbox):
    xmin, ymin, xmax, ymax = [int(b * 2) for b in bbox]
    w = xmax - xmin
    h = ymax - ymin
    p_h = h // 3
    p_w = w // 3
    crop = frame.crop((xmin, ymin, xmax, ymax))
    #crop = frame[max(ymin - p_h, 0):ymax + p_h, max(xmin - p_w, 0):xmax + p_w]
    #h, w = crop.shape[:2]
    return crop

def get_videos_from_folder(folder_path):
    # List all files in the specified folder
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    if 'metadata.json' in files:
        files.remove('metadata.json')
    full_paths = [os.path.join(folder_path, f) for f in files]
    return full_paths
def custom_collate_fn(batch):
    return batch
def process_videos(videos, detector=FacenetDetector()):
    dataset = VideoDataset(videos)
    loader = DataLoader(dataset, shuffle=False, batch_size=1, collate_fn=custom_collate_fn) # process video 1 by 1
    for item in tqdm(loader): # for each video
        faces = [] # faces in this video
        videos_faces = {}
        # num_workers=cpu_count() - 3
        video, indices, frames = item[0]
        batches = [frames[i:i + detector.batch_size] for i in range(0, len(frames), detector.batch_size)]
        for j, frames_batch in enumerate(batches): # iterate through batches of frames
            boxes_batch = detector.detect_faces(frames_batch) # detect batch of frames
            for i, bboxes in enumerate(boxes_batch): # only detect 1 img from box
                if bboxes is None:
                    continue
                for j, bbox in enumerate(bboxes):
                    if bbox is not None:
                        faces.append(crop_image(frames_batch[i], bbox))
            videos_faces[video] = faces
    return videos_faces
            
           # result.update({int(j * detector.batch_size) + i : b for i, b in zip(indices, detector.detect_faces(frames))})


