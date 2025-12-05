from typing import Literal

import torch
from torch import nn, optim
from torch.amp import autocast
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

import config
from utils import Timer, format_time, rgb2y


class Trainer:
    def __init__(
        self,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        model: nn.Module,
        loss_fn: nn.Module,
        optimizer: optim.Optimizer,
        epochs: int,
        scheduler: LRScheduler | None,
        device: Literal["cuda", "cpu"] = "cpu",
    ) -> None:
        self.device = device

        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.model = model.to(self.device)
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.epochs = epochs
        self.scheduler = scheduler

        self.psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(self.device)
        self.ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(
            self.device
        )
        self.timer = Timer(device=self.device)

        self.current_epoch = 0
        self.train_losses = []
        self.val_losses = []
        self.psnr_values = []
        self.ssim_values = []

    def _train_step(self) -> None:
        total_loss = 0.0

        self.model.train()

        for hr_img_tensor, lr_img_tensor in self.train_dataloader:
            hr_img_tensor = hr_img_tensor.to(self.device, non_blocking=True)
            lr_img_tensor = lr_img_tensor.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()

            with autocast(device_type=self.device, dtype=torch.bfloat16, enabled=True):
                sr_img_tensor = self.model(lr_img_tensor)
                loss = self.loss_fn(sr_img_tensor, hr_img_tensor)

            total_loss += loss.item()

            loss.backward()
            self.optimizer.step()

        self.train_losses.append(total_loss / len(self.train_dataloader))
        self.current_epoch += 1

    def _validation_step(self) -> None:
        total_loss = 0.0

        self.model.eval()

        with torch.inference_mode():
            for hr_img_tensor, lr_img_tensor in self.val_dataloader:
                hr_img_tensor = hr_img_tensor.to(self.device, non_blocking=True)
                lr_img_tensor = lr_img_tensor.to(self.device, non_blocking=True)

                with autocast(
                    device_type=self.device, dtype=torch.bfloat16, enabled=True
                ):
                    sr_img_tensor = self.model(lr_img_tensor)
                    loss = self.loss_fn(sr_img_tensor, hr_img_tensor)

                total_loss += loss.item()

                y_hr_img_tensor = rgb2y(hr_img_tensor.clamp(0.0, 1.0))
                y_sr_img_tensor = rgb2y(sr_img_tensor.clamp(0.0, 1.0))

                sf = config.SCALING_FACTOR
                y_hr_img_tensor = y_hr_img_tensor[:, :, sf:-sf, sf:-sf]
                y_sr_img_tensor = y_sr_img_tensor[:, :, sf:-sf, sf:-sf]

                self.psnr_metric.update(y_sr_img_tensor, y_hr_img_tensor)  # type: ignore
                self.ssim_metric.update(y_sr_img_tensor, y_hr_img_tensor)  # type: ignore

            self.val_losses.append(total_loss / len(self.val_dataloader))

            self.psnr_values.append(self.psnr_metric.compute().item())  # type: ignore
            self.ssim_values.append(self.ssim_metric.compute().item())  # type: ignore

            self.psnr_metric.reset()
            self.ssim_metric.reset()

    def train(self) -> None:
        elapsed_time_in_secs = 0.0

        for epoch in range(self.epochs):
            self.timer.start()

            self._train_step()
            self._validation_step()

            if self.scheduler:
                self.scheduler.step()

            epoch_duration_in_secs = self.timer.stop()
            elapsed_time_in_secs += epoch_duration_in_secs

            epoch_duration = format_time(epoch_duration_in_secs)
            elapsed_time = format_time(elapsed_time_in_secs)
            remaining_time = format_time(
                epoch_duration_in_secs * (self.epochs - self.current_epoch)
            )

            current_lr = self.optimizer.param_groups[0]["lr"]

            config.logger.info(
                f"Epoch: {self.current_epoch}/{self.epochs} ({epoch_duration} | {elapsed_time}/{remaining_time}) | LR: {current_lr:.2e} | Train loss: {self.train_losses[-1]:.4f} | Val loss: {self.val_losses[-1]:.4f} | PSNR: {self.psnr_values[-1]:.2f} | SSIM: {self.ssim_values[-1]:.2f}"
            )
