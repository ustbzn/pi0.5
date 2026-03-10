from typing import List

import torch

IMAGENET_STANDARD_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073])
IMAGENET_STANDARD_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711])


def add_image_tokens_to_prompt(
    prefix_prompt,
    bos_token,
    image_seq_len,
    image_token,
):
    return f"{image_token * image_seq_len}{prefix_prompt}"


def rescale(
    image: torch.LongTensor,
    scale: float,
) -> torch.FloatTensor:
    rescaled_image = image * scale
    return rescaled_image


def normalize(
    image: torch.LongTensor,
    mean: torch.FloatTensor,
    std: torch.FloatTensor,
) -> torch.FloatTensor:
    assert image.ndim == 5, f"Expected 5D tensor, got {image.ndim}D tensor."
    assert (
        image.shape[2] == 3
    ), f"Expected 3 channels at axis 1, got {image.shape[2]} channels."
    mean = mean[None, None, :, None, None]
    std = std[None, None, :, None, None]
    image = (image - mean) / std
    return image


def process_images(
    images: torch.LongTensor,
    rescale_factor: float,
    image_mean: torch.FloatTensor,
    image_std: torch.FloatTensor,
    max_pixels: int = 576 * 28 * 28,
    min_pixels: int = 16 * 28 * 28,
) -> torch.FloatTensor:
    """处理图像张量，支持自适应图像尺寸调整（官方Qwen3-VL格式）

    Args:
        images: 输入图像张量 [Batch_Size, Num_Images, Channels, Height, Width]
        rescale_factor: 缩放因子
        image_mean: 图像均值
        image_std: 图像标准差
        max_pixels: 最大像素数（官方默认: 576*28*28）
        min_pixels: 最小像素数（官方默认: 16*28*28）
    """
    # 自适应调整图像尺寸以符合像素限制
    batch_size, num_images, channels, h, w = images.shape
    current_pixels = h * w

    # 如果当前像素数超过最大值，进行缩放
    if current_pixels > max_pixels:
        scale_factor = (max_pixels / current_pixels) ** 0.5
        new_h = int(h * scale_factor)
        new_w = int(w * scale_factor)
        images = torch.nn.functional.interpolate(
            images.view(batch_size * num_images, channels, h, w),
            size=(new_h, new_w),
            mode='bilinear',
            align_corners=False
        ).view(batch_size, num_images, channels, new_h, new_w)

    # 如果当前像素数小于最小值，进行上采样
    elif current_pixels < min_pixels:
        scale_factor = (min_pixels / current_pixels) ** 0.5
        new_h = int(h * scale_factor)
        new_w = int(w * scale_factor)
        images = torch.nn.functional.interpolate(
            images.view(batch_size * num_images, channels, h, w),
            size=(new_h, new_w),
            mode='bilinear',
            align_corners=False
        ).view(batch_size, num_images, channels, new_h, new_w)

    images = rescale(images, scale=rescale_factor)
    images = normalize(images, mean=image_mean, std=image_std)
    return images


class QwenVLProcessor:
    IMAGE_TOKEN = "<image>"  # Qwen3-VL 的图像 token（官方格式）

    def __init__(
        self,
        tokenizer,
        num_image_tokens: int,
        max_seq_len: int,
        tokenizer_padding: str = "max_length",
        max_pixels: int = 576 * 28 * 28,
        min_pixels: int = 16 * 28 * 28,
    ):
        super().__init__()

        self.image_seq_length = num_image_tokens
        self.max_seq_len = max_seq_len
        self.tokenizer_padding = tokenizer_padding
        self.max_pixels = max_pixels
        self.min_pixels = min_pixels

        # 添加图像 token（官方格式）
        tokens_to_add = {"additional_special_tokens": [self.IMAGE_TOKEN]}
        tokenizer.add_special_tokens(tokens_to_add)
        self.image_token_id = tokenizer.convert_tokens_to_ids(self.IMAGE_TOKEN)

        self.tokenizer = tokenizer

    def __call__(
        self,
        text: List[str],
        images: torch.LongTensor,
        truncation: bool = True,
    ) -> dict:
        """处理文本和图像输入（官方Qwen3-VL格式）

        Args:
            text: 文本提示列表
            images: 图像张量 [Batch_Size, Num_Images, Channels, Height, Width]
            truncation: 是否截断过长的序列
        """
        assert len(images) == len(text), f"Received {len(images)} images for {len(text)} prompts."
        assert images.dtype == torch.uint8, f"Expected uint8 tensor for images, got {images.dtype}."

        pixel_values = process_images(
            images,
            rescale_factor=1 / 255.0,
            image_mean=IMAGENET_STANDARD_MEAN,
            image_std=IMAGENET_STANDARD_STD,
            max_pixels=self.max_pixels,
            min_pixels=self.min_pixels,
        )

        # Prepend image tokens to the prompt（官方格式：每个<image>对应一个图像）
        input_strings = [
            add_image_tokens_to_prompt(
                prefix_prompt=prompt,
                bos_token="",
                image_seq_len=self.image_seq_length,
                image_token=self.IMAGE_TOKEN,
            )
            for prompt in text
        ]

        inputs = self.tokenizer(
            input_strings,
            return_tensors="pt",
            max_length=self.max_seq_len,
            padding=self.tokenizer_padding,
            truncation=truncation,
        )
        output = {"pixel_values": pixel_values, **inputs}
        return output