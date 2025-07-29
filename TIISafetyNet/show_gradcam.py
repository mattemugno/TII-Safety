import argparse
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision.transforms import (
    Resize, CenterCrop, ToTensor, Normalize, Compose
)
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
        return out.logits


def reshape_transform(tensor, height=14, width=14):
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))
    result = result.permute(0, 3, 1, 2)
    return result


def visualize_cam_tensor(img_tensor, cam_map, mean, std):
    # De-normalizza e converte in immagine [H, W, C]
    img = (img_tensor * std[:, None, None] + mean[:, None, None])
    img = img.permute(1, 2, 0).cpu().numpy().clip(0, 1)
    return show_cam_on_image(img, cam_map, use_rgb=True)


def main(args):
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Carica modello e image processor
    hf_model = ViTForImageClassification.from_pretrained(
        args.model_path,
        num_labels=2,
        ignore_mismatched_sizes=True
    ).to(device)
    hf_model.eval()

    processor = ViTImageProcessor.from_pretrained(args.model_path)
    transform = Compose([
        Resize((processor.size['height'], processor.size['width'])),
        CenterCrop((processor.size['height'], processor.size['width'])),
        ToTensor(),
        Normalize(mean=processor.image_mean, std=processor.image_std),
    ])

    dataset = TIIDataset(args.data_root, transform=transform)

    # Split
    total = len(dataset)
    train_len = int(total * args.train_split)
    val_len = int(total * args.val_split)
    test_len = total - train_len - val_len

    _, _, test_ds = random_split(
        dataset,
        [train_len, val_len, test_len],
        generator=torch.Generator().manual_seed(args.seed)
    )

    def collate_fn(batch):
        return {
            'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
            'labels':       torch.tensor([x['label'] for x in batch])
        }

    loader = DataLoader(
        test_ds,
        batch_size=1,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available()
    )

    # GradCAM setup
    model = CAMWrapper(hf_model).to(device)
    target_layer = hf_model.vit.encoder.layer[-1].layernorm_before
    cam = GradCAM(model=model, target_layers=[target_layer], reshape_transform=reshape_transform)

    mean = torch.tensor(processor.image_mean).to(device)
    std = torch.tensor(processor.image_std).to(device)

    label_map = {0: "unsafe", 1: "safe"}

    # Grad-CAM per 9 immagini
    n_images = 9
    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    it = iter(loader)
    for idx in range(n_images):
        batch = next(it)
        pixels = batch['pixel_values'].to(device)
        label = int(batch['labels'][0].item())

        grayscale_cam = cam(
            input_tensor=pixels,
            targets=[ClassifierOutputTarget(label)]
        )[0]

        vis = visualize_cam_tensor(pixels[0], grayscale_cam, mean, std)

        ax = axes[idx // 3, idx % 3]
        ax.imshow(vis)
        ax.set_title(f"Label: {label_map[label]}")
        ax.axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Grad-CAM inference on ViT')
    parser.add_argument('--data-root',   type=str,   default='../data/dataset')
    parser.add_argument('--model-name',  type=str,   default='google/vit-base-patch16-224')
    parser.add_argument('--model-path',  type=str,   default='vit-base-glove')
    parser.add_argument('--train-split', type=float, default=0.7)
    parser.add_argument('--val-split',   type=float, default=0.15)
    parser.add_argument('--seed',        type=int,   default=2)
    parser.add_argument('--num-workers', type=int,   default=0)
    args = parser.parse_args()
    main(args)
