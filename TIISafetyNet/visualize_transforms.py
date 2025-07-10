import argparse
import random

import matplotlib.pyplot as plt
import torch
from torchvision.utils import make_grid
from torch.utils.data import DataLoader, Subset
from TIISafetyNet.load_dataset import get_transforms, TIIDataset


def denormalize(img_tensor):
    mean = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
    std = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
    return img_tensor * std + mean


def show_batch(images, labels, class_names):
    images = denormalize(images)
    grid_img = make_grid(images, nrow=4)
    np_img = grid_img.permute(1, 2, 0).numpy()

    plt.figure(figsize=(12, 6))
    plt.imshow(np_img)
    plt.axis('off')
    plt.title("Sample images with transforms")
    plt.show()

    print("Labels in batch:", [class_names[int(l)] for l in labels])

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
    transform, _ = get_transforms(args.blur)


    dataset = TIIDataset(args.data_root, transform=transform)
    subset = Subset(dataset, range(args.num_samples))
    loader = DataLoader(subset, batch_size=args.batch_size, shuffle=False)

    images, labels = next(iter(loader))
    #show_batch(images, labels, class_names=["no_glove", "glove"])
    show_one_image(images, labels, class_names=["No Glove", "Glove"])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize dataset transforms')
    parser.add_argument('--data-root', type=str, default='../data/dataset', help='Dataset root path')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size to visualize')
    parser.add_argument('--num-samples', type=int, default=32, help='Number of samples to load')
    parser.add_argument('--blur', default=True, action='store_true', help='Apply blur in transforms')
    args = parser.parse_args()

    main(args)
