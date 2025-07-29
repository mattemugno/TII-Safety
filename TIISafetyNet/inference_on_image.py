import argparse
from PIL import Image
import torch
from torchvision.transforms import (
    Resize, CenterCrop, ToTensor, Normalize, Compose, GaussianBlur
)
import matplotlib.pyplot as plt

from transformers import ViTForImageClassification, ViTImageProcessor


def apply_blur(image, kernel_size=7, sigma=2.0):
    blur = GaussianBlur(kernel_size=kernel_size, sigma=sigma)
    return blur(image)


def get_transform(image_processor):
    if "height" in image_processor.size:
        size = (image_processor.size["height"], image_processor.size["width"])
        crop_size = size
    elif "shortest_edge" in image_processor.size:
        size = image_processor.size["shortest_edge"]
        crop_size = (size, size)

    return Compose([
        Resize(size),
        CenterCrop(crop_size),
        ToTensor(),
        Normalize(mean=image_processor.image_mean, std=image_processor.image_std),
    ])


def predict(image, model_path, device):
    image_processor = ViTImageProcessor.from_pretrained(model_path)
    transform = get_transform(image_processor)

    image = transform(image).unsqueeze(0).to(device)

    model = ViTForImageClassification.from_pretrained(model_path)
    model.to(device)
    model.eval()

    with torch.no_grad():
        outputs = model(pixel_values=image)
        probs = torch.softmax(outputs.logits, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()
        conf = probs[0, pred_class].item()

    return pred_class, conf


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Caricamento e blur
    original_image = Image.open(args.image).convert('RGB')
    blurred_image = apply_blur(original_image, args.kernel_size, args.sigma)

    # Visualizza
    plt.imshow(blurred_image)
    plt.title(f"Blurred Input (k={args.kernel_size}, σ={args.sigma})")
    plt.axis("off")
    plt.show()

    # Inferenza
    pred_class, conf = predict(blurred_image, args.model, device)
    label_map = {0: "unsafe", 1: "safe"}
    label_str = f"Class {pred_class} ({label_map[pred_class]})"
    print(f"Prediction: {label_str} with confidence {conf:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ViT inference with Gaussian blur")
    parser.add_argument("--image", type=str, default='example_images/154370.015801041_5.jpeg', help="Path to image file")
    parser.add_argument("--model", type=str, default='vit-base-glove', help="Path to trained model directory")
    parser.add_argument("--kernel-size", type=int, default=7, help="Kernel size for Gaussian blur")
    parser.add_argument("--sigma", type=float, default=2.0, help="Sigma for Gaussian blur")
    args = parser.parse_args()
    main(args)
