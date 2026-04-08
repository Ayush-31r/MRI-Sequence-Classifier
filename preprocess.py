import numpy as np
import nibabel as nib
from PIL import Image
import torch
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

def preprocess(filepath):
    img  = nib.load(filepath)
    data = img.get_fdata()

    if data.ndim == 4:
        data = data[:, :, :, 0]

    mid = data.shape[2] // 2
    slice_2d = data[:, :, mid]

    mean = slice_2d.mean()
    std = slice_2d.std()
    slice_2d = (slice_2d - mean) / (std + 1e-8)

    slice_2d = np.clip(slice_2d, -3, 3)
    slice_2d = ((slice_2d + 3) / 6 * 255).astype(np.uint8)

    img_rgb = Image.fromarray(slice_2d).convert("RGB")
    tensor = transform(img_rgb).unsqueeze(0)  # (1, 3, 224, 224)
    return tensor