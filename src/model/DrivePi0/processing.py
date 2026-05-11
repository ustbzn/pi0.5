from typing import List, Optional

import torch

IMAGENET_STANDARD_MEAN = torch.tensor([0.5, 0.5, 0.5])
IMAGENET_STANDARD_STD = torch.tensor([0.5, 0.5, 0.5])


# 默认的 8 相机标签（4 方向 × 2 时间步），顺序与 train.py 中 image_list 拼接顺序一致
DEFAULT_CAM_LABELS = [
    'current front view', 'current right view',
    'current left view', 'current rear view',
    'previous front view', 'previous right view',
    'previous left view', 'previous rear view',
]


def add_image_tokens_to_prompt(
    prefix_prompt,
    bos_token,
    image_seq_len,
    image_token,
    cam_labels=None,
    num_images=1,
):
    """构建包含图像 token 和可选相机标签的 prompt。

    Args:
        prefix_prompt: 原始文本 prompt（如导航指令）
        bos_token: BOS token
        image_seq_len: 每张图的 token 数（如 256）
        image_token: image token 字符串（如 "<image>"）
        cam_labels: 每张图的文本标签列表，长度应等于 num_images。
                    为 None 时退回原始行为（所有 image token 连续拼接）
        num_images: 图像数量

    Returns:
        构建好的 prompt 字符串
    """
    if cam_labels is not None and len(cam_labels) == num_images:
        # 带标签模式：在每组 image token 前插入文本标签
        # 格式: "label1: <image>*256 label2: <image>*256 ... <bos> text\n"
        parts = []
        for label in cam_labels:
            parts.append(f"{label}: {image_token * image_seq_len} ")
        return f"{''.join(parts)}{bos_token}{prefix_prompt}\n"
    else:
        # 原始模式：所有 image token 连续拼接
        return f"{image_token * (image_seq_len * num_images)}{bos_token}{prefix_prompt}\n"


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
    mean = mean[None, None, :, None, None]  # add batch and spatial dimensions
    std = std[None, None, :, None, None]
    image = (image - mean) / std
    return image


def process_images(
    images: torch.LongTensor,# [btz, num_images, rgb_channels, h, w]
    rescale_factor: float,
    image_mean: torch.FloatTensor,
    image_std: torch.FloatTensor,
) -> torch.FloatTensor:
    # Rescale the pixel values to be in the range [0, 1]
    images = rescale(images, scale=rescale_factor)

    # Normalize the images to have mean 0 and standard deviation 1
    images = normalize(images, mean=image_mean, std=image_std)

    return images


class VLAProcessor:
    IMAGE_TOKEN = "<image>"

    def __init__(
        self,
        tokenizer,
        num_image_tokens: int,
        max_seq_len: int,
        tokenizer_padding: str = "max_length",  #  # instead of truncating to longest
        patches_per_image: int = 256,
    ):
        super().__init__()

        self.num_image_tokens = num_image_tokens  # 总 image token 数（如 2048 = 8 * 256）
        self.patches_per_image = patches_per_image  # 每张图的 patch 数（如 256）
        self.max_seq_len = max_seq_len
        self.tokenizer_padding = tokenizer_padding

        # Tokenizer described here: https://github.com/google-research/big_vision/blob/main/big_vision/configs/proj/paligemma/README.md#tokenizer
        tokens_to_add = {"additional_special_tokens": [self.IMAGE_TOKEN]}
        tokenizer.add_special_tokens(tokens_to_add)
        EXTRA_TOKENS = [
            f"<loc{i:04d}>" for i in range(1024)
        ]  # These tokens are used for object detection (bounding boxes)
        EXTRA_TOKENS += [
            f"<seg{i:03d}>" for i in range(128)
        ]  # These tokens are used for object segmentation
        tokenizer.add_tokens(EXTRA_TOKENS)
        self.image_token_id = tokenizer.convert_tokens_to_ids(self.IMAGE_TOKEN)
        # We will add the BOS and EOS tokens ourselves
        tokenizer.add_bos_token = False
        tokenizer.add_eos_token = False

        self.tokenizer = tokenizer

    def __call__(
        self,
        text: List[str],
        images: torch.LongTensor,
        cam_labels: Optional[List[str]] = None,
        truncation: bool = True,
    ) -> dict:
        assert len(images) == len(
            text
        ), f"Received {len(images)} images for {len(text)} prompts."
        # images: [btz, num_images, rgb_channels, h, w]
        # test: List [btz]
        assert (
            images.dtype == torch.uint8
        ), f"Expected uint8 tensor for images, got {images.dtype}."

        num_images = images.shape[1]

        pixel_values = process_images(
            images,
            rescale_factor=1 / 255.0,
            image_mean=IMAGENET_STANDARD_MEAN,
            image_std=IMAGENET_STANDARD_STD,
        ) # ->[btz, num_images, rgb_channels, h, w]

        # Prepend a `self.image_seq_length` number of image tokens to the prompt
        input_strings = [
            add_image_tokens_to_prompt(
                prefix_prompt=prompt,
                bos_token=self.tokenizer.bos_token,
                image_seq_len=self.patches_per_image,
                image_token=self.IMAGE_TOKEN,
                cam_labels=cam_labels,
                num_images=num_images,
            )
            for prompt in text
        ]

        # Returns the input_ids and attention_mask as PyTorch tensors
        inputs = self.tokenizer(
            input_strings,
            return_tensors="pt",
            max_length=self.max_seq_len,
            padding=self.tokenizer_padding,
            truncation=truncation,
        )
        output = {"pixel_values": pixel_values, **inputs}
        return output