from pathlib import Path
from typing import Literal

import torch
from safetensors.torch import load_file
from torch import nn
from torchvision.io import decode_image, write_png
from torchvision.transforms import v2 as transforms

import config
from models import ResidualChannelAttentionNetwork
from utils import compare_imgs, upscale_img_tiled


def inference(
    model: nn.Module,
    input_path: Path,
    output_path: Path,
    scaling_factor: Literal[2, 4, 8],
    use_downscale: bool,
    use_tiling: bool,
    create_comparisson: bool,
    comparisson_path: Path | None,
    orientation: Literal["vertical", "horizontal"],
    device: Literal["cuda", "cpu"] = "cpu",
) -> None:
    if not input_path.exists():
        raise FileNotFoundError(f"Input image not found: {input_path}")

    if input_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
        raise ValueError("Input image must be in JPG or PNG format")

    lr_img_tensor_uint8 = decode_image(str(input_path))

    original_lr_img_tensor_uint8 = lr_img_tensor_uint8

    if use_downscale:
        config.logger.info(f"Downscaling image by {scaling_factor} times...")

        _, lr_img_height, lr_img_width = lr_img_tensor_uint8.shape

        height_remainder = lr_img_height % scaling_factor
        width_remainder = lr_img_width % scaling_factor

        if height_remainder != 0 or width_remainder != 0:
            pad_top = height_remainder // 2
            pad_left = width_remainder // 2
            pad_bottom = pad_top + (lr_img_height - height_remainder)
            pad_right = pad_left + (lr_img_width - width_remainder)

            lr_img_tensor_uint8 = lr_img_tensor_uint8[
                :, pad_top:pad_bottom, pad_left:pad_right
            ]

        _, lr_img_height, lr_img_width = lr_img_tensor_uint8.shape

        lr_img_height_final = lr_img_height // scaling_factor
        lr_img_width_final = lr_img_width // scaling_factor

        lr_img_tensor_uint8 = transforms.Resize(
            size=(lr_img_height_final, lr_img_width_final),
            interpolation=transforms.InterpolationMode.BICUBIC,
            antialias=True,
        )(lr_img_tensor_uint8)

    lr_img_tensor = (
        (lr_img_tensor_uint8.to(torch.float32) / 255.0).unsqueeze(0).to(device)
    )

    if use_tiling:
        config.logger.info(
            f"Starting tiled inference with tile size: {config.TILE_SIZE} and tile overlap: {config.TILE_OVERLAP}..."
        )

        sr_img_tensor = upscale_img_tiled(
            model=model,
            lr_img_tensor=lr_img_tensor,
            scale_factor=scaling_factor,
            tile_size=config.TILE_SIZE,
            tile_overlap=config.TILE_OVERLAP,
            device=device,
        )
    else:
        with torch.inference_mode():
            sr_img_tensor = model(lr_img_tensor).cpu()

    if create_comparisson and comparisson_path:
        if comparisson_path.parent.exists():
            config.logger.info("Creating comparison image...")

            original_lr_img_tensor = (
                original_lr_img_tensor_uint8.to(torch.float32) / 255.0
            )

            compare_imgs(
                lr_img_tensor=lr_img_tensor,
                sr_img_tensor=sr_img_tensor,
                hr_img_tensor=original_lr_img_tensor
                if orientation == "horizontal"
                else None,
                output_path=comparisson_path,
                scaling_factor=scaling_factor,
                orientation=orientation,
            )
        else:
            config.logger.error("Comparison image path not found")
            raise FileNotFoundError

    sr_img_tensor_uint8 = (sr_img_tensor.clamp(0.0, 1.0) * 255.0).to(torch.uint8)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_png(sr_img_tensor_uint8.squeeze(0), str(output_path))

    config.logger.info(f"Upscaled image was saved to {output_path}")


if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = ResidualChannelAttentionNetwork(
        channels_count=config.CHANNELS_COUNT,
        kernel_size=config.KERNEL_SIZE,
        reduction=config.REDUCTION,
        rg_count=config.RESIDUAL_GROUPS_COUNT,
        rcab_count=config.RESIDUAL_CHANNEL_ATTENTION_BLOCKS_COUNT,
        scaling_factor=config.SCALING_FACTOR,
    ).to(device)

    if config.BEST_RCAN_CHECKPOINT_DIR_PATH.exists():
        model.load_state_dict(
            load_file(
                config.BEST_RCAN_CHECKPOINT_DIR_PATH / "model.safetensors",
                device=device,
            )
        )
    elif config.RCAN_CHECKPOINT_DIR_PATH.exists():
        model.load_state_dict(
            load_file(
                config.RCAN_CHECKPOINT_DIR_PATH / "model.safetensors",
                device=device,
            )
        )

    inference(
        model=model,
        input_path=config.INFERENCE_INPUT_IMG_PATH,
        output_path=config.INFERENCE_OUTPUT_IMG_PATH,
        scaling_factor=config.SCALING_FACTOR,
        use_downscale=True,
        use_tiling=False,
        create_comparisson=True,
        comparisson_path=config.INFERENCE_COMPARISON_IMG_PATH,
        orientation="horizontal",
        device=device,
    )
