import logging
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Literal

import torch
from torch import Tensor
from torchvision.io import decode_image
from torchvision.transforms import v2 as transforms


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

    if test_mode:
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

    normalize_transforms = transforms.Compose(
        [
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    hr_img_tensor = normalize_transforms(hr_img_tensor)
    lr_img_tensor = normalize_transforms(lr_img_tensor)

    return hr_img_tensor, lr_img_tensor


def rgb2y(img: Tensor) -> Tensor:
    ycbcr_weights_tensor = torch.tensor([0.257, 0.504, 0.098], device=img.device).view(
        1, 3, 1, 1
    )

    return torch.sum(img * ycbcr_weights_tensor, dim=1, keepdim=True) + 0.06
