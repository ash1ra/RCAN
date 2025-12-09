from typing import Literal

import torch
from safetensors.torch import load_file
from torch import nn
from torch.amp import autocast
from torch.utils.data import DataLoader
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

import config
from dataset import SRDataset
from models import ResidualChannelAttentionNetwork
from utils import rgb2y


def test_step(
    model: nn.Module,
    loss_fn: nn.Module,
    test_dataloader: DataLoader,
    psnr_metric: PeakSignalNoiseRatio,
    ssim_metric: StructuralSimilarityIndexMeasure,
    device: Literal["cuda", "cpu"] = "cpu",
) -> tuple[float, float, float]:
    total_loss = 0.0

    model.eval()

    with torch.inference_mode():
        for hr_img_tensor, lr_img_tensor in test_dataloader:
            hr_img_tensor = hr_img_tensor.to(device, non_blocking=True)
            lr_img_tensor = lr_img_tensor.to(device, non_blocking=True)

            with autocast(device_type=device, dtype=torch.bfloat16, enabled=True):
                sr_img_tensor = model(lr_img_tensor)
                loss = loss_fn(sr_img_tensor, hr_img_tensor)

            total_loss += loss.item()

            y_hr_img_tensor = rgb2y(hr_img_tensor.clamp(0.0, 1.0))
            y_sr_img_tensor = rgb2y(sr_img_tensor.clamp(0.0, 1.0))

            sf = config.SCALING_FACTOR
            y_hr_img_tensor = y_hr_img_tensor[:, :, sf:-sf, sf:-sf]
            y_sr_img_tensor = y_sr_img_tensor[:, :, sf:-sf, sf:-sf]

            psnr_metric.update(y_sr_img_tensor, y_hr_img_tensor)  # type: ignore
            ssim_metric.update(y_sr_img_tensor, y_hr_img_tensor)  # type: ignore

        test_loss = total_loss / len(test_dataloader)

        psnr_value = psnr_metric.compute().item()  # type: ignore
        ssim_value = ssim_metric.compute().item()  # type: ignore

        psnr_metric.reset()
        ssim_metric.reset()

    return test_loss, psnr_value, ssim_value


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

    loss_fn = nn.L1Loss()

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

    for dataset_path in config.TEST_DATASET_PATHS:
        test_dataset = SRDataset(
            data_path=dataset_path,
            scaling_factor=config.SCALING_FACTOR,
            crop_size=config.CROP_SIZE,
            test_mode=True,
        )

        test_dataloader = DataLoader(
            dataset=test_dataset,
            batch_size=config.TEST_BATCH_SIZE,
            shuffle=False,
            num_workers=config.VALIDATION_NUM_WORKERS,
            pin_memory=True if device == "cuda" else False,
        )

        test_loss, test_psnr, test_ssim = test_step(
            model=model,
            loss_fn=loss_fn,
            test_dataloader=test_dataloader,
            psnr_metric=PeakSignalNoiseRatio(data_range=1.0).to(device),
            ssim_metric=StructuralSimilarityIndexMeasure(data_range=1.0).to(device),
            device=device,
        )

        config.logger.info(
            f"{dataset_path.name} Dataset | Test loss: {test_loss:.4f} | PSNR: {test_psnr:.4f} | SSIM: {test_ssim:.4f}"
        )
