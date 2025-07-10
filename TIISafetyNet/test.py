import argparse
import torch
from torch.utils.data import DataLoader, random_split
from transformers import ViTForImageClassification
from sklearn.metrics import classification_report

from TIISafetyNet.load_dataset import get_transforms, TIIDataset


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images).logits
        preds = torch.argmax(outputs, dim=1)

        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    return all_preds, all_labels


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load transforms
    _, val_transform = get_transforms(blur=False)

    # Load dataset (transform is applied only after splitting)
    full_dataset = TIIDataset(args.data_root, transform=None, collapse=args.collapse)
    total = len(full_dataset)
    train_size = int(total * args.train_split)
    val_size = int(total * args.val_split)
    test_size = total - train_size - val_size

    # Split dataset
    _, _, test_ds = random_split(
        full_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(args.seed)
    )
    test_ds.dataset.transform = val_transform

    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # Load model
    model = ViTForImageClassification.from_pretrained(
        'google/vit-base-patch16-224',
        num_labels=2,
        ignore_mismatched_sizes=True
    )
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)

    print(f"Evaluating model on {test_size} test samples...")
    preds, labels = evaluate(model, test_loader, device)

    # Report results
    acc = sum(p == l for p, l in zip(preds, labels)) / len(labels)
    print(f"Test Accuracy: {acc:.4f}")
    print("Classification Report:")
    print(classification_report(labels, preds, target_names=['no-glove', 'glove']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate trained ViT model on test set")
    parser.add_argument('--data-root', type=str, default='../data/dataset', help='Root path to dataset')
    parser.add_argument('--model-path', type=str, default='models/vit_glove.pth', help='Path to trained model weights (.pth)')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for test loader')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of workers for DataLoader')
    parser.add_argument('--collapse', type=str, choices=['any', 'left', 'right', 'both'], default='both',
                        help='How to collapse glove labels')
    parser.add_argument('--train-split', type=float, default=0.7, help='Train split ratio (for reproducibility)')
    parser.add_argument('--val-split', type=float, default=0.15, help='Val split ratio (for reproducibility)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for splitting')
    args = parser.parse_args()

    main(args)
