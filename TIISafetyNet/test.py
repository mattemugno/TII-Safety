import argparse
import torch
from torch.utils.data import DataLoader, random_split
from transformers import ViTForImageClassification, ViTImageProcessor
from sklearn.metrics import classification_report

from TIISafetyNet.load_dataset import TIIDataset

def collate_fn(batch):
    return {
        'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
        'labels': torch.tensor([x['labels'] for x in batch])
    }

def evaluate(model, test_ds, device, batch_size=32, num_workers=4):
    model.eval()
    all_preds = []
    all_labels = []

    loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate_fn,     # la tua fn che impila pixel_values e labels
        pin_memory=torch.cuda.is_available()
    )

    with torch.no_grad():
        for batch in loader:
            pixels = batch['pixel_values'].to(device)  # [B,C,H,W]
            labels = batch['labels'].to(device)        # [B]

            logits = model(pixels).logits              # [B, num_labels]
            preds = torch.argmax(logits, dim=1)        # [B]

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    return all_preds, all_labels


def main(args):
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    processor = ViTImageProcessor.from_pretrained(args.model_name)
    transform = lambda img: processor(img, return_tensors='pt')['pixel_values'].squeeze(0)

    # Load dataset with transform
    dataset = TIIDataset(args.data_root, transform=transform)
    total = len(dataset)
    train_len = int(total * args.train_split)
    val_len = int(total * args.val_split)
    test_len = total - train_len - val_len

    _, _, test_ds = random_split(
        dataset, [train_len, val_len, test_len],
        generator=torch.Generator().manual_seed(args.seed)
    )

    model = ViTForImageClassification.from_pretrained(
        args.model_path,
        num_labels=2,
        ignore_mismatched_sizes=True
    )
    model.to(device)

    print(f"Evaluating on {test_len} samples...")
    preds, labels = evaluate(model, test_ds, device)

    acc = sum(p == l for p, l in zip(preds, labels)) / len(labels)
    print(f"Test Accuracy: {acc:.4f}")
    print("Classification Report:")
    class_names = ['safe', 'unsafe']
    print(classification_report(labels, preds, target_names=class_names))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate ViT model (3 classes)')
    parser.add_argument('--data-root', type=str, default='../data/dataset', help='Dataset root')
    parser.add_argument('--model-name', type=str, default='google/vit-base-patch16-224',
                        help='Pretrained ViT identifier or path')
    parser.add_argument('--model-path', type=str, default='vit-base-glove', help='Path to .pth weights')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--train-split', type=float, default=0.7)
    parser.add_argument('--val-split', type=float, default=0.15)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    main(args)
