import logging
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from time import perf_counter
from typing import Literal

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from torch import Tensor, nn
from torchvision.io import decode_image
from torchvision.transforms import v2 as transforms


class Timer:
    def __init__(self, device: Literal["cuda", "cpu"]) -> None:
        self.is_cuda = True if device == "cuda" else False

        if self.is_cuda:
            self.start_event = torch.cuda.Event(enable_timing=True)
            self.end_event = torch.cuda.Event(enable_timing=True)
        else:
            self.start_time = 0.0

    def start(self) -> None:
        if self.is_cuda:
            self.start_event.record()
        else:
            self.start_time = perf_counter()

    def stop(self) -> float:
        if self.is_cuda:
            self.end_event.record()
            torch.cuda.synchronize()

            return self.start_event.elapsed_time(self.end_event) / 1000
        else:
            return perf_counter() - self.start_time


def create_logger(
    log_level: str,
    log_file_name: str,
    max_log_file_size: int = 5 * 1024 * 1024,
    backup_count: int = 10,
) -> logging.Logger:
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y.%m.%d %H:%M:%S"
    )

    logger.handlers.clear()

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(getattr(logging, log_level.upper()))
    logger.addHandler(console_handler)

    current_date = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    log_file_name = f"logs/{log_file_name}_{current_date}.log"

    log_file_path = Path(log_file_name)
    log_file_path.parent.mkdir(parents=True, exist_ok=True)

    file_handler = RotatingFileHandler(
        filename=log_file_path,
        maxBytes=max_log_file_size,
        backupCount=backup_count,
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def create_hr_and_lr_imgs(
    img_path: str | Path,
    scaling_factor: Literal[2, 4, 8],
    crop_size: int | None = None,
    test_mode: bool = False,
) -> tuple[Tensor, Tensor]:
    img_tensor = decode_image(Path(img_path).__fspath__())

    img_tensor = transforms.ToDtype(torch.float32, scale=True)(img_tensor)

    if test_mode:
        _, height, width = img_tensor.shape

        height_remainder = height % scaling_factor
        width_remainder = width % scaling_factor

        top_bound = height_remainder // 2
        left_bound = width_remainder // 2

        bottom_bound = top_bound + (height - height_remainder)
        right_bound = left_bound + (width - width_remainder)

        hr_img_tensor = img_tensor[:, top_bound:bottom_bound, left_bound:right_bound]
    elif crop_size:
        augmentation_transforms = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomChoice(
                    [
                        transforms.RandomRotation(degrees=[0, 0]),
                        transforms.RandomRotation(degrees=[90, 90]),
                        transforms.RandomRotation(degrees=[180, 180]),
                        transforms.RandomRotation(degrees=[270, 270]),
                    ]
                ),
                transforms.RandomCrop(size=(crop_size, crop_size)),
            ]
        )

        hr_img_tensor = augmentation_transforms(img_tensor)

    lr_transforms = transforms.Compose(
        [
            transforms.Resize(
                size=(
                    hr_img_tensor.shape[1] // scaling_factor,
                    hr_img_tensor.shape[2] // scaling_factor,
                ),
                interpolation=transforms.InterpolationMode.BICUBIC,
                antialias=True,
            )
        ]
    )

    lr_img_tensor = lr_transforms(hr_img_tensor)

    return hr_img_tensor, lr_img_tensor


def compare_imgs(
    lr_img_tensor: Tensor,
    sr_img_tensor: Tensor,
    output_path: str | Path,
    hr_img_tensor: Tensor | None = None,
    scaling_factor: Literal[2, 4, 8] = 4,
    orientation: Literal["horizontal", "vertical"] = "vertical",
) -> None:
    bicubic_label = "Bicubic"
    sr_label = "RCAN"
    hr_label = "Original"

    to_pil_img_transform = transforms.ToPILImage()

    lr_img = to_pil_img_transform(lr_img_tensor)
    sr_img = to_pil_img_transform(sr_img_tensor)

    bicubic_img = transforms.Resize(
        size=(sr_img_tensor.shape[2], sr_img_tensor.shape[3]),
        interpolation=transforms.InterpolationMode.BICUBIC,
    )(lr_img)

    width, height = sr_img.size

    if orientation == "horizontal" and isinstance(hr_img_tensor, Tensor):
        hr_img = to_pil_img_transform(hr_img_tensor)

        total_width = width * 3 + 50
        total_height = height

        comparison_img = Image.new("RGB", (total_width, total_height), color="white")

        comparison_img.paste(bicubic_img, (0, 50))
        comparison_img.paste(sr_img, (width + 25, 50))
        comparison_img.paste(hr_img, (width * 2 + 50, 50))
    else:
        total_width = width
        total_height = height * 2 + 100

        comparison_img = Image.new("RGB", (total_width, total_height), color="white")

        comparison_img.paste(bicubic_img, (0, 50))
        comparison_img.paste(sr_img, (0, height + 100))

    draw = ImageDraw.Draw((comparison_img))

    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/TTF/JetBrainsMonoNerdFont-Regular.ttf", size=36
        )
    except OSError:
        font = ImageFont.load_default()

    bicubic_text_width = draw.textlength(bicubic_label, font=font)
    sr_text_width = draw.textlength(sr_label, font=font)
    hr_text_width = draw.textlength(hr_label, font=font)

    if orientation == "horizontal":
        draw.text(
            ((width - bicubic_text_width) / 2, 5),
            bicubic_label,
            fill="black",
            font=font,
        )

        draw.text(
            ((width - sr_text_width) / 2 + width + 25, 5),
            sr_label,
            fill="black",
            font=font,
        )

        draw.text(
            ((width - hr_text_width) / 2 + width * 2 + 50, 5),
            hr_label,
            fill="black",
            font=font,
        )
    else:
        draw.text(
            ((width - bicubic_text_width) / 2, 5),
            bicubic_label,
            fill="black",
            font=font,
        )

        draw.text(
            ((width - sr_text_width) / 2, height + 55),
            sr_label,
            fill="black",
            font=font,
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    comparison_img.save(output_path, format="PNG")


def upscale_img_tiled(
    model: nn.Module,
    lr_img_tensor: Tensor,
    scale_factor: Literal[2, 4, 8] = 4,
    tile_size: int = 512,
    tile_overlap: int = 64,
    device: Literal["cuda", "cpu"] = "cpu",
) -> Tensor:
    batch_size, channels, height_original, width_original = lr_img_tensor.shape

    height_target = height_original * scale_factor
    width_target = width_original * scale_factor

    border_pad = tile_overlap // 2

    lr_img_tensor_padded = F.pad(
        lr_img_tensor, (border_pad, border_pad, border_pad, border_pad), "reflect"
    )

    _, _, height_padded, width_padded = lr_img_tensor_padded.shape

    step_size = tile_size - tile_overlap

    pad_right = (step_size - (width_padded - tile_size) % step_size) % step_size
    pad_bottom = (step_size - (height_padded - tile_size) % step_size) % step_size

    lr_img_tensor_padded = F.pad(
        lr_img_tensor_padded, (0, pad_right, 0, pad_bottom), "reflect"
    )

    _, _, height_final, width_final = lr_img_tensor_padded.shape

    final_img_canvas = torch.zeros(
        (batch_size, channels, height_final * scale_factor, width_final * scale_factor),
        dtype=lr_img_tensor.dtype,
        device="cpu",
    )

    count_canvas = torch.zeros_like(final_img_canvas, device="cpu")

    for height in range(0, height_final - tile_size + 1, step_size):
        for width in range(0, width_final - tile_size + 1, step_size):
            lr_img_tensor_tile = lr_img_tensor_padded[
                :, :, height : height + tile_size, width : width + tile_size
            ].to(device, non_blocking=True)

            with torch.inference_mode():
                sr_img_tensor_tile = model(lr_img_tensor_tile).cpu()

            final_height_start = height * scale_factor
            final_width_start = width * scale_factor
            final_height_end = (height + tile_size) * scale_factor
            final_width_end = (width + tile_size) * scale_factor

            final_img_canvas[
                :,
                :,
                final_height_start:final_height_end,
                final_width_start:final_width_end,
            ] += sr_img_tensor_tile

            count_canvas[
                :,
                :,
                final_height_start:final_height_end,
                final_width_start:final_width_end,
            ] += 1

    output_padded = final_img_canvas / count_canvas

    final_border_pad = border_pad * scale_factor

    final_output = output_padded[
        :,
        :,
        final_border_pad : final_border_pad + height_target,
        final_border_pad : final_border_pad + width_target,
    ]

    return final_output


def rgb2y(img: Tensor) -> Tensor:
    ycbcr_weights_tensor = torch.tensor([0.257, 0.504, 0.098], device=img.device).view(
        1, 3, 1, 1
    )

    return torch.sum(img * ycbcr_weights_tensor, dim=1, keepdim=True) + 0.06


def format_time(total_seconds: float) -> str:
    if total_seconds < 0:
        total_seconds = 0

    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)

    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
