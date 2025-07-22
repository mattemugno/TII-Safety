import argparse
from PIL import Image
import torch
from torchvision.transforms import GaussianBlur
import matplotlib.pyplot as plt

from transformers import AutoImageProcessor, ViTForImageClassification


def apply_blur(image, kernel_size=7, sigma=2.0):
    blur = GaussianBlur(kernel_size=kernel_size, sigma=sigma)
    return blur(image)


def predict(image, model_path, device):
    # Load processor and model
    processor = AutoImageProcessor.from_pretrained('google/vit-base-patch16-224')
    model = ViTForImageClassification.from_pretrained(
        'google/vit-base-patch16-224',
        num_labels=2,
        ignore_mismatched_sizes=True
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Preprocess image (pass as list!)
    inputs = processor(images=[image], return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()
        conf = probs[0, pred_class].item()

    return pred_class, conf


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load and blur image
    original_image = Image.open(args.image).convert('RGB')
    blurred_image = apply_blur(original_image, args.kernel_size, args.sigma)

    # Show blurred image
    plt.imshow(blurred_image)
    plt.title(f"Blurred Input (k={args.kernel_size}, σ={args.sigma})")
    plt.axis("off")
    plt.show()

    # Inference
    pred_class, conf = predict(blurred_image, args.model, device)
    label_str = f"Class {pred_class} ({'Glove' if pred_class == 1 else 'No Glove'})"
    print(f"Prediction: {label_str} with confidence {conf:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ViT inference with Gaussian blur")
    parser.add_argument("--image", type=str, default='example_images/154370.015801041_5.jpeg', help="Path to image file")
    parser.add_argument("--model", type=str, default='vit-base-glove', help="Path to trained model (.pth)")
    parser.add_argument("--kernel-size", type=int, default=7, help="Kernel size for Gaussian blur")
    parser.add_argument("--sigma", type=float, default=2.0, help="Sigma for Gaussian blur")
    args = parser.parse_args()
    main(args)
