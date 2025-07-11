import argparse
from collections import Counter

import numpy as np
import torch
from evaluate import load
from torch.utils.data import random_split
from transformers import ViTForImageClassification, TrainingArguments, Trainer, ViTImageProcessor

from TIISafetyNet.load_dataset import TIIDataset


def collate_fn(batch):
    return {
        'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
        'labels': torch.tensor([x['labels'] for x in batch])
    }


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return load('accuracy').compute(predictions=preds, references=labels)


def main(args):
    # Initialize processor
    processor = ViTImageProcessor.from_pretrained(args.model_name)
    transform = lambda img: processor(img, return_tensors='pt')['pixel_values'].squeeze(0)

    # Load dataset with transform
    dataset = TIIDataset(args.data_root, transform=transform)
    total = len(dataset)
    train_len = int(total * args.train_split)
    val_len = int(total * args.val_split)
    test_len = total - train_len - val_len

    train_ds, val_ds, test_ds = random_split(
        dataset, [train_len, val_len, test_len],
        generator=torch.Generator().manual_seed(args.seed)
    )

    # Print overview
    labels = [lbl for _, lbl in dataset.samples]
    counter = Counter(labels)
    print(f"Total samples: {total}")
    print(f"Class distribution -> unsafe (0): {counter.get(0, 0)}, safe (1): {counter.get(1, 0)}")
    print(f"Splits (train/val/test): {train_len}/{val_len}/{test_len}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    model = ViTForImageClassification.from_pretrained(
        args.model_name, num_labels=2, ignore_mismatched_sizes=True
    )

    model.to(device)
    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        eval_strategy='steps',
        eval_steps=args.eval_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        learning_rate=args.lr,
        fp16=torch.cuda.is_available(),
        remove_unused_columns=False,
        load_best_model_at_end=True,
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
        tokenizer=processor,
    )

    # Train & evaluate
    train_results = trainer.train()
    trainer.log_metrics("train", train_results.metrics)
    trainer.save_metrics("train", train_results.metrics)
    trainer.save_model()
    print("Training completed.")

    metrics = trainer.evaluate(val_ds)
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=str, default='../data/dataset')
    parser.add_argument('--model-name', type=str, default='google/vit-base-patch16-224-in21k')
    parser.add_argument('--output-dir', type=str, default='vit-base-glove')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--train-split', type=float, default=0.7)
    parser.add_argument('--val-split', type=float, default=0.15)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--epochs', type=int, default=4)
    parser.add_argument('--logging-steps', type=int, default=10)
    parser.add_argument('--eval-steps', type=int, default=100)
    parser.add_argument('--save-steps', type=int, default=100)
    args = parser.parse_args()
    main(args)

