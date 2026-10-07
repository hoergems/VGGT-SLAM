"""SAM3-specific helpers owned by the VGGT-SLAM integration."""

import torch


def run_sam3_text_query(processor, image, query):
    """Run one SAM3 image/text query with CUDA BF16 autocast when available.

    SAM3's image and text stages share state, so both calls must be in the
    same autocast context. CPU inference intentionally keeps its normal
    precision because CUDA autocast is not available there.
    """
    if torch.cuda.is_available():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            state = processor.set_image(image)
            return processor.set_text_prompt(state=state, prompt=query)

    state = processor.set_image(image)
    return processor.set_text_prompt(state=state, prompt=query)
