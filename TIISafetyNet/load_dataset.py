import json
import os

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import copy
from torch.utils.data import random_split

class GaussianBlur:
    def __init__(self, ksize=(33, 33)):
        self.ksize = ksize

    def __call__(self, img):
        if not isinstance(img, Image.Image):
            raise TypeError("Expected PIL.Image")

        img_np = np.array(img)
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        blurred = cv2.GaussianBlur(img_np, self.ksize, 1)
        blurred = cv2.cvtColor(blurred, cv2.COLOR_BGR2RGB)
        return Image.fromarray(blurred)


class TIIDataset(Dataset):
    """
    Dataset that reads images and JSON annotations recursively from a root folder.
    Assumes each JSON file has a 'label' key with 0 or 1, and corresponding image with same basename + .jpeg.
    """

    def __init__(self, root_dir: str, transform=None):
        self.transform = transform
        self.blur = GaussianBlur()
        self.samples = []

        for subj_folder in os.listdir(root_dir):
            if not subj_folder.endswith('gloves'):
                continue
            subj_path = os.path.join(root_dir, subj_folder)
            subj_name = subj_folder.split()[0]

            ann_path = os.path.join(subj_path, f"annotations_{subj_name}.json")
            if not os.path.exists(ann_path):
                continue
            with open(ann_path, 'r') as jf:
                ann_data = json.load(jf)

            for session, file_dict in ann_data.items():
                data_dir = os.path.join(subj_path, session, 'data')
                if not os.path.isdir(data_dir):
                    continue

                for fname, label_str in file_dict.items():

                    img_path = os.path.join(data_dir, fname)
                    if not os.path.exists(img_path):
                        continue

                    try:
                        label = int(label_str)
                    except ValueError:
                        continue

                    self.samples.append((img_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')

        image = self.blur(image)
        if self.transform:
            image = self.transform(image)
        return {'pixel_values': image, 'label': label}

    def split(self, train_frac, val_frac, seed=None):
        """
        Splits this dataset into three TIIDataset objects (train/val/test).
        Each returned object is a shallow copy of self but with .samples
        filtered to only the corresponding indices.
        """
        total = len(self)
        train_len = int(total * train_frac)
        val_len = int(total * val_frac)
        test_len = total - train_len - val_len

        # random_split gives Subsets, but we only care about their indices
        subsets = random_split(
            self,
            [train_len, val_len, test_len],
            generator=torch.Generator().manual_seed(seed or 0)
        )

        def make_clone(subset):
            ds = copy.copy(self)
            ds.samples = [self.samples[i] for i in subset.indices]
            return ds

        train_ds = make_clone(subsets[0])
        val_ds = make_clone(subsets[1])
        test_ds = make_clone(subsets[2])
        return train_ds, val_ds, test_ds