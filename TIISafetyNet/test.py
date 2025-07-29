import argparse

import torch
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    Resize,
    ToTensor,
)
from transformers import ViTForImageClassification, ViTImageProcessor

from TIISafetyNet.load_dataset import TIIDataset


def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}

def evaluate(model, test_ds, device, batch_size=32, num_workers=4):
    model.eval()
    all_preds = []
    all_labels = []

    loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available()
    )

    with torch.no_grad():
        for batch in loader:
            pixels = batch['pixel_values'].to(device)
            labels = batch['labels'].to(device)

            logits = model(pixels).logits
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    return all_preds, all_labels


def main(args):
    image_processor = ViTImageProcessor.from_pretrained(args.model_name)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = TIIDataset(args.data_root)
    total = len(dataset)
    train_len = int(total * args.train_split)
    val_len = int(total * args.val_split)
    test_len = total - train_len - val_len

    _, _, test_ds = dataset.split(
        train_frac=args.train_split,
        val_frac=args.val_split,
        seed=args.seed
    )

    normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
    if "height" in image_processor.size:
        size = (image_processor.size["height"], image_processor.size["width"])
        crop_size = size
        max_size = None
    elif "shortest_edge" in image_processor.size:
        size = image_processor.size["shortest_edge"]
        crop_size = (size, size)
        max_size = image_processor.size.get("longest_edge")

    val_transforms = Compose(
        [
            Resize(size),
            CenterCrop(crop_size),
            ToTensor(),
            normalize,
        ]
    )

    test_ds.transform = val_transforms

    model = ViTForImageClassification.from_pretrained(
        args.model_path,
        label2id={'unsafe':0, 'safe':1},
        id2label={0:'unsafe', 1:'safe'},
        num_labels=2,
        ignore_mismatched_sizes=True
    )
    model.to(device)

    print(f"Evaluating on {test_len} samples...")
    preds, labels = evaluate(model, test_ds, device)

    acc = sum(p == l for p, l in zip(preds, labels)) / len(labels)
    print(f"Test Accuracy: {acc:.4f}")
    print("Classification Report:")
    class_names = ['unsafe', 'safe']
    print(classification_report(labels, preds, target_names=class_names))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate ViT model (2 classes)')
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
