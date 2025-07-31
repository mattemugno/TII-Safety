from typing import List, Optional, Callable

import numpy as np
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image


class HuggingfaceToTensorModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super(HuggingfaceToTensorModelWrapper, self).__init__()
        self.model = model

    def forward(self, x):
        return self.model(x).logits


def category_name_to_index(model, category_name):
    name_to_index = dict((v, k) for k, v in model.config.id2label.items())
    return name_to_index[category_name]

def run_grad_cam_on_image(
    model: torch.nn.Module,
    target_layer: torch.nn.Module,
    targets_for_gradcam: List[Callable],
    reshape_transform: Optional[Callable],
    input_tensor: torch.Tensor,
    input_image: np.ndarray,
    method: Callable = GradCAM,
    eigen_smooth: bool = False,
    aug_smooth: bool = False
):
    wrapper = HuggingfaceToTensorModelWrapper(model)

    with method(
        model=wrapper,
        target_layers=[target_layer],
        reshape_transform=reshape_transform
    ) as cam:
        cam.batch_size = 1

        visualizations = []
        for target in targets_for_gradcam:
            grayscale_cam = cam(
                input_tensor=input_tensor.unsqueeze(0),
                targets=[target],
                eigen_smooth=eigen_smooth,
                aug_smooth=aug_smooth
            )[0]

            vis = show_cam_on_image(
                np.array(input_image).astype(np.float32) / 255.0,
                grayscale_cam,
                use_rgb=True
            )
            visualizations.append(vis)
        return np.hstack(visualizations)