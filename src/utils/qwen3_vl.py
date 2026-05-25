import logging


log = logging.getLogger(__name__)


def sync_qwen3_video_processor(processor):
    """Keep Qwen3-VL video processor patch settings aligned with image processor.

    Some transformers versions load `image_processor.temporal_patch_size` from the
    pretrained directory but leave `video_processor.temporal_patch_size` at the
    class default. That breaks video patch packing when the pretrained config uses
    a non-default temporal patch size such as 3.
    """

    video_processor = getattr(processor, "video_processor", None)
    image_processor = getattr(processor, "image_processor", None)
    if video_processor is None or image_processor is None:
        return processor

    synced = {}
    for attr in ("patch_size", "temporal_patch_size", "merge_size"):
        image_value = getattr(image_processor, attr, None)
        video_value = getattr(video_processor, attr, None)
        if image_value is None:
            continue
        if video_value != image_value:
            setattr(video_processor, attr, image_value)
            synced[attr] = (video_value, image_value)

    if synced:
        log.warning("Synced Qwen3VL video processor settings from image processor: %s", synced)

    return processor
