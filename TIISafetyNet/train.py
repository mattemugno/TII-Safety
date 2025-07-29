import argparse
from collections import Counter

import evaluate
import numpy as np
import torch
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
)
from transformers import ViTForImageClassification, TrainingArguments, Trainer, ViTImageProcessor

from TIISafetyNet.load_dataset import TIIDataset

metric = evaluate.load("accuracy")


def compute_metrics(eval_pred):
    """Computes accuracy on a batch of predictions"""
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return metric.compute(predictions=predictions, references=eval_pred.label_ids)


def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}


def main(args):
    image_processor = ViTImageProcessor.from_pretrained(args.model_name)

    dataset = TIIDataset(args.data_root)
    total = len(dataset)
    train_len = int(total * args.train_split)
    val_len = int(total * args.val_split)
    test_len = total - train_len - val_len

    train_ds, val_ds, test_ds = dataset.split(
        train_frac=args.train_split,
        val_frac=args.val_split,
        seed=args.seed
    )

    labels = sorted({label for _, label in train_ds.samples})
    label2id = {str(label): idx for idx, label in enumerate(labels)}
    id2label = {idx: str(label) for idx, label in enumerate(labels)}

    normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
    if "height" in image_processor.size:
        size = (image_processor.size["height"], image_processor.size["width"])
        crop_size = size
        max_size = None
    elif "shortest_edge" in image_processor.size:
        size = image_processor.size["shortest_edge"]
        crop_size = (size, size)
        max_size = image_processor.size.get("longest_edge")

    train_transforms = Compose(
        [
            RandomResizedCrop(crop_size),
            RandomHorizontalFlip(),
            ToTensor(),
            normalize,
        ]
    )

    val_transforms = Compose(
        [
            Resize(size),
            CenterCrop(crop_size),
            ToTensor(),
            normalize,
        ]
    )

    train_ds.transform = train_transforms
    val_ds.transform = val_transforms
    test_ds.transform = val_transforms

    labels = [lbl for _, lbl in dataset.samples]
    counter = Counter(labels)
    print(f"Total samples: {total}")
    print(f"Class distribution -> unsafe (0): {counter.get(0, 0)}, safe (1): {counter.get(1, 0)}")
    print(f"Splits (train/val/test): {train_len}/{val_len}/{test_len}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = ViTForImageClassification.from_pretrained(
        args.model_name,
        label2id=label2id,
        id2label=id2label,
        ignore_mismatched_sizes=True
    )

    model.to(device)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        remove_unused_columns=False,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        dataloader_num_workers=args.num_workers,
        warmup_ratio=0.1,
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
        tokenizer=image_processor,
    )

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
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--epochs', type=int, default=4)
    parser.add_argument('--logging-steps', type=int, default=10)
    parser.add_argument('--eval-steps', type=int, default=100)
    parser.add_argument('--save-steps', type=int, default=100)
    args = parser.parse_args()
    main(args)
