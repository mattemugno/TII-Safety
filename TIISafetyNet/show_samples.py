import argparse
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.utils import make_grid
from torch.utils.data import DataLoader, Subset
from transformers import ViTImageProcessor

from TIISafetyNet.load_dataset import TIIDataset


def denormalize(img_tensor):
    mean = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
    std = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
    return img_tensor * std + mean


def show_batch(images, labels, class_names):
    images = denormalize(images)
    batch_size = images.size(0)
    n_cols = 3
    n_rows = int(np.ceil(batch_size / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4))
    axes = axes.flatten()

    for i in range(len(images)):
        img = images[i].permute(1, 2, 0).numpy()
        axes[i].imshow(img)
        axes[i].axis('off')
        axes[i].set_title(class_names[int(labels[i])])

    for j in range(len(images), len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()

def show_one_image(images, labels, class_names):
    images = denormalize(images)
    idx = random.randint(0, len(images) - 1)
    img = images[idx]
    label = labels[idx]

    np_img = img.permute(1, 2, 0).numpy()

    plt.figure(figsize=(4, 4))
    plt.imshow(np_img)
    plt.axis('off')
    plt.title(f"Label: {class_names[int(label)]}")
    plt.show()


def main(args):
    random.seed(args.seed)

    processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
    transform = lambda img: processor(img, return_tensors='pt')['pixel_values'].squeeze(0)

    dataset = TIIDataset(args.data_root, transform=transform)

    zeros = [i for i, (_, lab) in enumerate(dataset.samples) if lab == 0]
    ones  = [i for i, (_, lab) in enumerate(dataset.samples) if lab == 1]

    if len(zeros) < 5 or len(ones) < 4:
        raise ValueError(f"Non ci sono abbastanza esempi nel dataset: zeros={len(zeros)}, ones={len(ones)}")

    sel_zeros = random.sample(zeros, 5)
    sel_ones  = random.sample(ones,  4)
    selected_idx = sel_zeros + sel_ones
    random.shuffle(selected_idx)

    images_list = []
    labels_list = []
    for idx in selected_idx:
        sample = dataset[idx]  # {'pixel_values': tensor, 'labels': int}
        images_list.append(sample['pixel_values'])
        labels_list.append(sample['labels'])

    images = torch.stack(images_list, dim=0)
    labels = torch.tensor(labels_list, dtype=torch.long)

    #show_batch(images, labels, class_names=["Unsafe", "Safe"])
    show_one_image(images, labels, class_names=["Unsafe", "Safe"])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize dataset transforms')
    parser.add_argument('--data-root', type=str, default='../data/dataset', help='Dataset root path')
    parser.add_argument('--batch-size', type=int, default=9, help='Batch size to visualize')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for sampling')
    args = parser.parse_args()

    main(args)
