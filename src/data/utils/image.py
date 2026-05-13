import torch
import numpy as np
import tensorflow as tf
import torch.nn.functional as F
import torchvision.transforms as transforms
from typing import Tuple
from PIL import Image, ImageDraw

from src.data.utils.augmentations import augment_image

CAM_VISUALIZE_MAP = {
    0: 0,
    1: 2,
    2: 4,
    3: 3,
    4: 5
}

def process_image(image: torch.Tensor) -> torch.Tensor:
    """Process the image by rounding, clipping, and casting to uint8."""
    # Ensure the image is in float32 before processing (if it's in a different dtype)
    image = image.float()

    # Round the image values to nearest integer
    image = torch.round(image)

    # Clip the values to ensure they are within the range [0, 255]
    image = torch.clamp(image, 0, 255)

    # Convert the image back to uint8 type
    image = image.to(torch.uint8)

    return image

def resize_image_tensor(image: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
    """Resize a tensor image using bilinear interpolation (or other methods)."""
    # Ensure the image is in the correct shape [batch_size, channels, height, width]
    image = process_image(image)
    image = image.unsqueeze(0)  # Add batch dimension if needed
    resized_image = F.interpolate(image, size=size, mode='bilinear', align_corners=False)
    return resized_image.squeeze(0)  # Remove the batch dimension


def read_resize_encode_image_pytorch(path: str, size: Tuple[int, int]) -> bytes:
    """Reads, decodes, resizes, and then re-encodes an image using PyTorch."""
    image = Image.open(path)
    
    resize_transform = transforms.Resize(size)
    image = resize_transform(image)
    
    image = image.convert("RGB")
    image = torch.tensor(np.array(image), dtype=torch.float32)
    image = torch.round(image)
    image = torch.clamp(image, 0, 255)
    image = image.to(torch.uint8)

    return image

def image_normalization(path: str, size: Tuple[int, int]):
    image = read_resize_encode_image_pytorch(path, size)
    np_array = image.numpy()
    tf_tensor = tf.convert_to_tensor(np_array)
    image_augment_kwargs = dict(random_resized_crop=dict(scale=[0.8, 1.0], ratio=[0.9, 1.1],),
                                random_brightness=[0.1],
                                random_contrast=[0.9, 1.1],
                                random_saturation=[0.9, 1.1],
                                random_hue=[0.05],
                                augment_order=[
                                    "random_resized_crop",
                                    "random_brightness",
                                    "random_contrast",
                                    "random_saturation",
                                    "random_hue",
                                ],
                            )
    output = augment_image(image=tf_tensor, **image_augment_kwargs)
    np_image = output.numpy().astype(np.uint8)
    image = torch.as_tensor(np_image)
    return image

def mosaic_driver_cameras(image_list, cam_id):
    if len(image_list) != 6:
        raise ValueError("6 images are needed")
    single_w, single_h = image_list[0].size
    total_w = single_w * 3
    total_h = single_h * 2
    mosaic_img = Image.new('RGB', (total_w, total_h), (0, 0, 0))
    
    mosaic_img.paste(image_list[0], (0, 0))
    mosaic_img.paste(image_list[1], (single_w, 0))
    mosaic_img.paste(image_list[2], (single_w * 2, 0))
    mosaic_img.paste(image_list[3], (0, single_h))
    mosaic_img.paste(image_list[4], (single_w, single_h))
    mosaic_img.paste(image_list[5], (single_w * 2, single_h))
    
    return mosaic_img

def mosaic_driver_cameras(image_list, cam_id):
    if len(image_list) != 6:
        raise ValueError("6 images are needed")
    
    single_w, single_h = image_list[0].size
    total_w = single_w * 3
    total_h = single_h * 2
    mosaic_img = Image.new('RGB', (total_w, total_h), (0, 0, 0))
    positions = [
        (0, 0), (single_w, 0), (single_w * 2, 0),
        (0, single_h), (single_w, single_h), (single_w * 2, single_h)
    ]
    
    for img, pos in zip(image_list, positions):
        mosaic_img.paste(img, pos)
    
    cam_id = CAM_VISUALIZE_MAP.get(cam_id)
    if 0 <= cam_id < len(positions):
        draw = ImageDraw.Draw(mosaic_img)
        x0, y0 = positions[cam_id]
        x1, y1 = x0 + single_w, y0 + single_h
        draw.rectangle([x0, y0, x1, y1], outline="red", width=4)

    return mosaic_img