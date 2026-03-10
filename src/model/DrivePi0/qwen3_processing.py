from PIL import Image
import torch
from transformers import AutoProcessor

def to_pil(img: torch.Tensor) -> Image.Image:
    # img: [C, H, W], uint8
    if img.ndim == 3 and img.shape[0] in (1, 3):
        img = img.permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(img)

class DriveQwen3Processor:
    def __init__(self, model_name_or_path: str, max_seq_len: int | None = None, padding: str = "longest"):
        self.processor = AutoProcessor.from_pretrained(model_name_or_path)
        self.max_seq_len = max_seq_len
        self.padding = padding

    def _build_prompt(self, text: str, num_images: int) -> str:
        content = [{"type": "image"} for _ in range(num_images)]
        content.append({"type": "text", "text": text})
        messages = [{"role": "user", "content": content}]
        return self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

    def __call__(self, text: list[str], images: torch.Tensor):
        # images: [B, N, C, H, W], uint8
        batch_images = []
        for sample in images:
            batch_images.append([to_pil(x) for x in sample])

        prompts = [self._build_prompt(t, len(imgs)) for t, imgs in zip(text, batch_images)]

        outputs = self.processor(
            text=prompts,
            images=batch_images,
            padding=self.padding,
            truncation=True,
            max_length=self.max_seq_len,
            return_tensors="pt",
        )
        return outputs