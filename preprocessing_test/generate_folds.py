import argparse
import json
import os
import random
from functools import partial
from multiprocessing.pool import Pool
from pathlib import Path

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import pandas as pd

from tqdm import tqdm

from os_helper import *

import cv2

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)


def get_paths(vid, label, root_dir):
    ori_vid, fake_vid = vid
    ori_dir = os.path.join(root_dir, "crops", ori_vid)
    fake_dir = os.path.join(root_dir, "crops", fake_vid)
    data = []
    for frame in range(320):
        if frame % 10 != 0:
            continue
        for actor in range(2): # this was for the diffs, not needed
            image_id = "{}_{}.png".format(frame, actor) # this was for the diffs, not needed
            ori_img_path = os.path.join(ori_dir, image_id)
            fake_img_path = os.path.join(fake_dir, image_id)
            img_path = ori_img_path if label == 0 else fake_img_path # where is the image located based on label
            try:
                # img = cv2.imread(img_path)[..., ::-1]
                if os.path.exists(img_path):
                    data.append([img_path, label, ori_vid])
            except:
                pass
    return data


root_dir = os.getcwd()
n_splits = 16
out = 'folds.csv'
seed=0
def main():
    ori_fakes = get_original_with_fakes(root_dir)
    sz = 50 // n_splits
    folds = []
    for fold in range(n_splits):
        folds.append(list(range(sz * fold, sz * fold + sz if fold < n_splits - 1 else 50)))
    print(folds)
    video_fold = {}
    for d in os.listdir(root_dir):
        if "dfdc" in d:
            part = int(d.split("_")[-1])
            for f in os.listdir(os.path.join(root_dir, d)):
                if "metadata.json" in f:
                    with open(os.path.join(root_dir, d, "metadata.json")) as metadata_json:
                        metadata = json.load(metadata_json)

                    for k, v in metadata.items():
                        fold = None
                        for i, fold_dirs in enumerate(folds):
                            if part in fold_dirs:
                                fold = i
                                break
                        assert fold is not None
                        video_id = k[:-4]
                        video_fold[video_id] = fold
    for fold in range(len(folds)):
        holdoutset = {k for k, v in video_fold.items() if v == fold}
        trainset = {k for k, v in video_fold.items() if v != fold}
        assert holdoutset.isdisjoint(trainset), "Folds have leaks"
    data = []
    ori_ori = set([(ori, ori) for ori, fake in ori_fakes])
    with Pool(processes=os.cpu_count()) as p:
        with tqdm(total=len(ori_ori)) as pbar:
            func = partial(get_paths, label=0, root_dir=root_dir)
            for v in p.imap_unordered(func, ori_ori):
                pbar.update()
                data.extend(v)
        with tqdm(total=len(ori_fakes)) as pbar:
            func = partial(get_paths, label=1, root_dir=root_dir)
            for v in p.imap_unordered(func, ori_fakes):
                pbar.update()
                data.extend(v)
    fold_data = []
    for img_path, label, ori_vid in data:
        path = Path(img_path)
        video = path.parent.name
        file = path.name
        assert video_fold[video] == video_fold[ori_vid], "original video and fake have leak  {} {}".format(ori_vid,
                                                                                                           video)
        fold_data.append([video, file, label, ori_vid, int(file.split("_")[0]), video_fold[video]])
    random.shuffle(fold_data)
    pd.DataFrame(fold_data, columns=["video", "file", "label", "original", "frame", "fold"]).to_csv(out, index=False)


if __name__ == '__main__':
    main()
