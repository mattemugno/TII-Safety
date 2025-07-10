import argparse

import torch
from torch import optim
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from transformers import ViTForImageClassification
from torch.utils.data import WeightedRandomSampler
from TIISafetyNet.load_dataset import get_transforms, TIIDataset


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for images, labels in tqdm(loader, desc='Training'):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images).logits
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def eval_epoch(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in tqdm(loader, desc='Validation'):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images).logits
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total if total > 0 else 0

def main(args):
    train_transform, val_transform = get_transforms(args.blur)

    # load full dataset
    full_dataset = TIIDataset(args.data_root, transform=None)
    total = len(full_dataset)
    train_size = int(total * args.train_split)
    val_size = int(total * args.val_split)
    test_size = total - train_size - val_size

    train_ds, val_ds, test_ds = random_split(full_dataset, [train_size, val_size, test_size],
                                             generator=torch.Generator().manual_seed(args.seed))

    from collections import Counter
    labels = [label for _, label in full_dataset.samples]
    counter = Counter(labels)
    print(f"Total samples: {total}")
    print(f"Class distribution -> no-glove (0): {counter.get(0, 0)}, glove (1): {counter.get(1, 0)}")

    # assign transforms
    train_ds.dataset.transform = train_transform
    val_ds.dataset.transform = val_transform
    test_ds.dataset.transform = val_transform

    # DataLoaders with oversampling for minority class
    train_labels = [train_ds.dataset.samples[i][1] for i in train_ds.indices]
    label_counts = Counter(train_labels)
    # Inverse frequency weights
    weights = [1.0 / label_counts[label] for label in train_labels]
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    # loaders
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size,
                             shuffle=False, num_workers=args.num_workers)

    print(f"Total samples: {total}")
    print(f"Train/Val/Test splits: {train_size}/{val_size}/{test_size}")
    print(f"Batches per split: {len(train_loader)}/{len(val_loader)}/{len(test_loader)}")

    # Example iteration
    imgs, labels = next(iter(train_loader))
    print("Batch img tensor shape", imgs.shape)
    print("Batch labels", labels.shape)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize ViT model
    model = ViTForImageClassification.from_pretrained(
        'google/vit-base-patch16-224',
        num_labels=2,
        ignore_mismatched_sizes=True
    )
    model.to(device)

    # Optimizer and loss
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    criterion = CrossEntropyLoss()

    best_val_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_acc = eval_epoch(model, val_loader, device)
        print(f"Epoch {epoch}/{args.epochs} - Loss: {train_loss:.4f}, Val Acc: {val_acc:.4f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), args.save_path)
            print(f"Saved best model with Val Acc: {val_acc:.4f}")

    # Final test evaluation
    test_acc = eval_epoch(model, test_loader, device)
    print(f"Test Accuracy: {test_acc:.4f}")



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='PyTorch DataLoader for glove/no-glove dataset')
    parser.add_argument('--data-root', type=str, default='../data/dataset',
                        help='Root directory containing subfolders with JSON and images')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--num-workers', type=int, default=8,
                        help='DataLoader worker count')
    parser.add_argument('--blur', default=True, action='store_true',
                        help='Apply random Gaussian blur on training images')
    parser.add_argument('--train-split', type=float, default=0.7,
                        help='Fraction of data for training')
    parser.add_argument('--val-split', type=float, default=0.15,
                        help='Fraction of data for validation')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for splitting')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--save-path', type=str, default='models/vit_glove.pth',
                        help='Path to save best model')
    args = parser.parse_args()
    main(args)
