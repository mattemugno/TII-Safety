import os
import json
import argparse
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms


def get_transforms(blur: bool = True):
    """
    Returns training and validation transforms.
    If blur=True, applies a random Gaussian blur on training images.
    """
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_ops = []
    if blur:
        train_ops.append(transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=5, sigma=(1.0, 3.0))
        ], p=0.8))

    train_ops += [
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ]

    val_ops = [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize
    ]

    return transforms.Compose(train_ops), transforms.Compose(val_ops)


class TIIDataset(Dataset):
    """
    Dataset that reads images and JSON annotations recursively from a root folder.
    Assumes each JSON file has a 'label' key with 0 or 1, and corresponding image with same basename + .jpeg.
    """

    def __init__(self, root_dir: str, transform=None, collapse: str = 'both'):
        self.transform = transform
        self.collapse = collapse
        self.samples = []

        # Iterate subjects
        for subj in os.listdir(root_dir):
            subj_path = os.path.join(root_dir, subj)
            if not os.path.isdir(subj_path):
                continue
            # Load annotation file for this subject
            ann_file = os.path.join(subj_path, f"annotations_{subj.split(' ')[0]}.json")
            if not os.path.exists(ann_file):
                continue
            with open(ann_file, 'r') as jf:
                ann_data = json.load(jf)

            # Iterate sessions
            for session in os.listdir(subj_path):
                sess_path = os.path.join(subj_path, session)
                data_dir = os.path.join(sess_path, 'data')
                if not os.path.isdir(data_dir):
                    continue
                # Collect images
                for fname in os.listdir(data_dir):
                    if not fname.lower().endswith(('.jpeg', '.jpg', '.png')):
                        continue
                    img_path = os.path.join(data_dir, fname)
                    # Lookup annotations by basename
                    entry = ann_data.get(fname)
                    if entry is None:
                        continue
                    # Expect entry like [0, 1] or {'left':0,'right':1}
                    if isinstance(entry, list) and len(entry) == 2:
                        left_label, right_label = int(entry[0]), int(entry[1])
                    elif isinstance(entry, dict) and 'left' in entry and 'right' in entry:
                        left_label, right_label = int(entry['left']), int(entry['right'])
                    else:
                        continue

                    # Collapse labels if needed
                    if self.collapse == 'any':
                        label = int(bool(left_label or right_label))
                    elif self.collapse == 'left':
                        label = left_label
                    elif self.collapse == 'right':
                        label = right_label
                    elif self.collapse == 'both':
                        label = int(bool(left_label and right_label))
                    else:
                        raise ValueError(f"Unknown collapse mode: {self.collapse}")
                    self.samples.append((img_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label
