import argparse
import warnings
from functools import partial

from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

from TIISafetyNet.explainability_utils import category_name_to_index, run_grad_cam_on_image

warnings.filterwarnings('ignore')
from pytorch_grad_cam import run_dff_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from PIL import Image
import torch
from transformers import AutoImageProcessor, \
    AutoModelForImageClassification

def load_and_preprocess(image_path, processor):
    image = Image.open(image_path).convert('RGB')

    transforms = Compose([
        Resize((224, 224)),
        CenterCrop((224, 224)),
        ToTensor(),
        Normalize(mean=processor.image_mean, std=processor.image_std),
    ])
    tensor_image = transforms(image)
    return tensor_image, image.resize((224, 224))


def main(args):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    processor = AutoImageProcessor.from_pretrained(args.model_name, use_fast=True)
    model = AutoModelForImageClassification.from_pretrained(
        args.model_path,
        num_labels=2,
        ignore_mismatched_sizes=True
    ).to(device).eval()

    input_tensor, image = load_and_preprocess(args.image, processor)
    input_tensor = input_tensor.to(device)

    def swinT_reshape_transform_huggingface(tensor, width, height):
        result = tensor.reshape(tensor.size(0),
                                height,
                                width,
                                tensor.size(2))
        result = result.transpose(2, 3).transpose(1, 2)
        return result

    reshape_transform = partial(swinT_reshape_transform_huggingface,
                                width=input_tensor.shape[2] // 32,
                                height=input_tensor.shape[1] // 32)

    targets_for_gradcam = [ClassifierOutputTarget(category_name_to_index(model, "0")),
                           ClassifierOutputTarget(category_name_to_index(model, "1"))]

    target_layer = model.swin.layernorm

    if args.method == "gradcam":
        cam_result = run_grad_cam_on_image(
            model=model,
            target_layer=target_layer,
            targets_for_gradcam=targets_for_gradcam,
            input_tensor=input_tensor,
            input_image=image,
            reshape_transform=reshape_transform
        )

        Image.fromarray(cam_result).save('gradcam_swin_result.jpeg', format='JPEG')

    elif args.method == "dff":
        dff_result = run_dff_on_image(model=model.cpu(),
            target_layer=target_layer,
            classifier=model.classifier,
            img_pil=image,
            img_tensor=input_tensor.cpu(),
            reshape_transform=reshape_transform,
            n_components=2,
            top_k=2)

        Image.fromarray(dff_result).save('dff_swin_result.jpeg', format='JPEG')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Grad-CAM per ViT HuggingFace")
    parser.add_argument('--image', type=str, default='example_images/154370.015801041_5.jpeg')
    parser.add_argument('--model-name', type=str, default='microsoft/swin-tiny-patch4-window7-224')
    parser.add_argument('--model-path', type=str, default='swin-tiny-glove')
    parser.add_argument('--eigen_smooth', action='store_true', help='Reduce noise by taking the first principle componenet of cam_weights*activations')
    parser.add_argument('--aug_smooth', action='store_true', help='Apply test time augmentation to smooth the CAM')
    parser.add_argument('--method', type=str, default='gradcam', help='Can be gradcam/gradcam++/scorecam/xgradcam/ablationcam')
    args = parser.parse_args()
    main(args)
