import random
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from transformers import ViTForImageClassification

from TIISafetyNet.load_dataset import get_transforms, TIIDataset

class ViTWrapper(nn.Module):
    def __init__(self, clf_model: ViTForImageClassification):
        super().__init__()
        self.clf = clf_model
        self.vit_model = clf_model.vit

    def forward(self, x):
        return self.clf(x).logits

def reshape_transform(tensor, height=14, width=14):
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))
    result = result.transpose(2, 3).transpose(1, 2)
    return result

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
raw_vit = ViTForImageClassification.from_pretrained(
    'google/vit-base-patch16-224',
    num_labels=2,
    ignore_mismatched_sizes=True
)

raw_vit.load_state_dict(torch.load('models/vit_glove.pth', map_location=device))
raw_vit.to(device).eval()
model = ViTWrapper(raw_vit).to(device)

target_layer = model.vit_model.encoder.layer[-1].layernorm_before
cam = GradCAM(
    model=model,
    target_layers=[target_layer],
    reshape_transform=reshape_transform
)

transform, _ = get_transforms(blur=False)
test_ds = TIIDataset(root_dir='../data/dataset', transform=transform)

idx = random.randrange(len(test_ds))
img_t, label = test_ds[idx]
inp = img_t.unsqueeze(0).to(device)
targets = [ClassifierOutputTarget(int(label))]

grayscale_cam = cam(input_tensor=inp, targets=targets)[0, :]

mean = torch.tensor([0.485, 0.456, 0.406], device=device)
std  = torch.tensor([0.229, 0.224, 0.225], device=device)
img_np = (img_t.to(device) * std[:, None, None] + mean[:, None, None])\
         .permute(1,2,0).cpu().numpy().clip(0,1)

vis = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)

plt.imshow(vis)
plt.title(f'Label: {label}')
plt.axis('off')
plt.show()
