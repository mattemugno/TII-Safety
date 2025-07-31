import argparse
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from TIISafetyNet.load_dataset import TIIDataset
from transformers import AutoImageProcessor
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize


def denormalize(img_tensor):
    mean = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
    std = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
    return img_tensor * std + mean


def show_batch(images, labels, class_names):
    images = denormalize(images)
    batch_size = images.size(0)
    n_cols = 2
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


def main(args):
    random.seed(args.seed)

    processor = AutoImageProcessor.from_pretrained(args.model_name)

    normalize = Normalize(mean=processor.image_mean, std=processor.image_std)
    if "height" in processor.size:
        size = (processor.size["height"], processor.size["width"])
        crop_size = size
    else:
        size = processor.size["shortest_edge"]
        crop_size = (size, size)

    val_transforms = Compose([
        Resize(size),
        CenterCrop(crop_size),
        ToTensor(),
        normalize,
    ])

    dataset = TIIDataset(args.data_root)
    dataset.transform = val_transforms

    zeros = [i for i, (_, lab) in enumerate(dataset.samples) if lab == 0]
    ones = [i for i, (_, lab) in enumerate(dataset.samples) if lab == 1]

    if len(zeros) < 5 or len(ones) < 4:
        raise ValueError("Not enough samples: zeros={}, ones={}".format(len(zeros), len(ones)))

    sel_zeros = random.sample(zeros, 2)
    sel_ones = random.sample(ones, 2)
    selected_idx = sel_zeros + sel_ones
    random.shuffle(selected_idx)

    images = []
    labels = []
    for idx in selected_idx:
        sample = dataset[idx]
        images.append(sample["pixel_values"])
        labels.append(sample["label"])

    images = torch.stack(images, dim=0)
    labels = torch.tensor(labels, dtype=torch.long)

    show_batch(images, labels, class_names=["Unsafe", "Safe"])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Visualize dataset");
    parser.add_argument('--data-root', type=str, default='../data/dataset')
    parser.add_argument('--model-name', type=str, default='facebook/deit-tiny-patch16-224')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    main(args)
