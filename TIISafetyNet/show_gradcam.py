import argparse

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torch.utils.data import DataLoader, random_split
from transformers import ViTForImageClassification, ViTImageProcessor

from TIISafetyNet.load_dataset import TIIDataset


class CAMWrapper(nn.Module):
    def __init__(self, hf_model: ViTForImageClassification):
        super().__init__()
        self.model = hf_model

    def forward(self, x: torch.Tensor):
        out = self.model(pixel_values=x)
        assert hasattr(out, "logits") and isinstance(out.logits, torch.Tensor), \
            f"Expected logits Tensor, got {type(out)}"
        return out.logits


def reshape_transform(tensor, height=14, width=14):
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))
    result = result.transpose(2, 3).transpose(1, 2)
    return result


def visualize_cam_tensor(img_tensor, cam_map, mean, std):
    img = (img_tensor * std[:, None, None] + mean[:, None, None]) \
          .permute(1, 2, 0).cpu().numpy().clip(0, 1)
    return show_cam_on_image(img, cam_map, use_rgb=True)


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    hf_model = ViTForImageClassification.from_pretrained(
        args.model_path,
        num_labels=2,
        ignore_mismatched_sizes=True
    ).to(device)
    hf_model.eval()

    model = CAMWrapper(hf_model).to(device)
    processor = ViTImageProcessor.from_pretrained(args.model_name)

    transform = lambda img: processor(img, return_tensors='pt')['pixel_values'].squeeze(0)
    dataset = TIIDataset(args.data_root, transform=transform)
    total = len(dataset)
    train_len = int(total * args.train_split)
    val_len   = int(total * args.val_split)
    test_len  = total - train_len - val_len

    _, _, test_ds = random_split(
        dataset,
        [train_len, val_len, test_len],
        generator=torch.Generator().manual_seed(args.seed)
    )

    def collate_fn(batch):
        return {
            'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
            'labels':        torch.tensor([x['labels'] for x in batch])
        }

    loader = DataLoader(
        test_ds,
        batch_size=1,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available()
    )

    target_layer = hf_model.vit.encoder.layer[-1].layernorm_before
    cam = GradCAM(
        model=model,
        target_layers=[target_layer],
        reshape_transform=reshape_transform
    )

    mean = torch.tensor([0.485, 0.456, 0.406], device=device)
    std  = torch.tensor([0.229, 0.224, 0.225], device=device)

    batch = next(iter(loader))
    pixels = batch['pixel_values']
    labels = batch['labels']
    # debug rapido
    print("pixels type:", type(pixels), "labels type:", type(labels))

    pixels = pixels.to(device)
    lbl = int(labels[0].item())

    grayscale_cam = cam(
        input_tensor=pixels,
        targets=[ClassifierOutputTarget(lbl)]
    )[0]  # H x W

    vis = visualize_cam_tensor(pixels[0], grayscale_cam, mean, std)
    plt.figure(figsize=(5,5))
    plt.imshow(vis)
    plt.title(f'Label: {lbl}')
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Grad-CAM inference on ViT')
    parser.add_argument('--data-root',   type=str,   default='../data/dataset',
                        help='Dataset root folder')
    parser.add_argument('--model-name',  type=str,   default='google/vit-base-patch16-224',
                        help='Name of the transformer model')
    parser.add_argument('--model-path',  type=str,   default='vit-base-glove',
                        help='Fine-tuned model folder.')
    parser.add_argument('--train-split', type=float, default=0.7)
    parser.add_argument('--val-split',   type=float, default=0.15)
    parser.add_argument('--seed',        type=int,   default=42)
    parser.add_argument('--num-workers', type=int,   default=4)
    args = parser.parse_args()
    main(args)
