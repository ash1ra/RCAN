from datetime import datetime
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import seaborn as sns
import torch
from safetensors.torch import load_file, save_file
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

    def save_checkpoint(self, checkpoint_path: Path) -> None:
        checkpoint_path.mkdir(parents=True, exist_ok=True)

        save_file(self.model.state_dict(), checkpoint_path / "model.safetensors")

        state_dict = {
            "optimizer_state_dict": self.optimizer.state_dict(),
            "current_epoch": self.current_epoch,
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "psnr_values": self.psnr_values,
            "ssim_values": self.ssim_values,
        }

        if self.scheduler:
            state_dict["scheduler_state_dict"] = self.scheduler.state_dict()

        torch.save(state_dict, checkpoint_path / "learning_state.pt")

        config.logger.debug(
            f'Checkpoint was saved to "{checkpoint_path}" after {self.current_epoch} epochs'
        )

    def load_checkpoint(self, checkpoint_path: Path) -> None:
        if checkpoint_path.exists():
            if (checkpoint_path / "model.safetensors").exists():
                self.model.load_state_dict(
                    load_file(checkpoint_path / "model.safetensors", device=self.device)
                )
            else:
                config.logger.error(
                    f"File {checkpoint_path / 'model.safetensors'} not found"
                )
                raise FileNotFoundError

            if (checkpoint_path / "learning_state.pt").exists():
                state_dict = torch.load(
                    checkpoint_path / "learning_state.pt", map_location=self.device
                )

                self.optimizer.load_state_dict(state_dict["optimizer_state_dict"])

                if self.scheduler:
                    self.scheduler.load_state_dict(state_dict["scheduler_state_dict"])

                self.current_epoch = state_dict["current_epoch"]
                self.train_losses = state_dict["train_losses"]
                self.val_losses = state_dict["val_losses"]
                self.psnr_values = state_dict["psnr_values"]
                self.ssim_values = state_dict["ssim_values"]
            else:
                config.logger.error(
                    f"File {checkpoint_path / 'learning_state.pt'} not found"
                )
                raise FileNotFoundError

            config.logger.info(
                f"Checkpoint from {self.current_epoch} epoch was successfully loaded"
            )
        else:
            config.logger.error(f"Directory {checkpoint_path} not found")
            raise FileNotFoundError

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
        best_psnr = float("-inf")

        try:
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

                if self.current_epoch % config.CHECKPOINT_SAVING_FREQUENCY == 0:
                    self.save_checkpoint(
                        Path(
                            f"{config.RCAN_CHECKPOINT_DIR_PATH_TEMPLATE}_epoch_{self.current_epoch}"
                        )
                    )

                if self.psnr_values[-1] > best_psnr:
                    best_psnr = self.psnr_values[-1]
                    self.save_checkpoint(config.BEST_RCAN_CHECKPOINT_DIR_PATH)

            self.plot()

        except KeyboardInterrupt:
            config.logger.info("Saving model's weights and finish training...")
            self.save_checkpoint(
                Path(
                    f"{config.RCAN_CHECKPOINT_DIR_PATH_TEMPLATE}_epoch_{self.current_epoch}"
                )
            )

            self.plot()

    def plot(self) -> None:
        sns.set_style("whitegrid")
        sns.set_palette("deep")
        palette = sns.color_palette("deep")

        epochs = list(range(1, self.current_epoch + 1))

        fig, axs = plt.subplots(2, 2, figsize=(14, 10))

        fig.suptitle("RCAN Training Metrics", fontsize=18)

        hyperparameters_str = f"Scaling factor: {config.SCALING_FACTOR} | Crop size: {config.CROP_SIZE} | Batch size: {config.TRAIN_BATCH_SIZE} | Learning rate: {config.LEARNING_RATE} | Epochs: {config.EPOCHS} | Number of workers: {config.TRAIN_NUM_WORKERS} | Dev mode: {config.DEV_MODE}"

        fig.text(0.5, 0.94, hyperparameters_str, ha="center", va="top", fontsize=10)

        sns.lineplot(
            x=epochs,
            y=self.train_losses,
            ax=axs[0, 0],
            linewidth=2.5,
            color=palette[0],
        )

        axs[0, 0].set_title("Training loss")
        axs[0, 0].set_xlabel("Epoch")
        axs[0, 0].set_ylabel("Loss")

        sns.lineplot(
            x=epochs,
            y=self.val_losses,
            ax=axs[0, 1],
            linewidth=2.5,
            color=palette[1],
        )

        axs[0, 1].set_title("Validation loss")
        axs[0, 1].set_xlabel("Epoch")
        axs[0, 1].set_ylabel("Loss")

        sns.lineplot(
            x=epochs,
            y=self.psnr_values,
            ax=axs[1, 0],
            linewidth=2.5,
            color=palette[1],
        )
        axs[1, 0].set_title("Validation Peak Signal-to-Noise Ratio")
        axs[1, 0].set_xlabel("Epoch")
        axs[1, 0].set_ylabel("PSNR")

        sns.lineplot(
            x=epochs,
            y=self.ssim_values,
            ax=axs[1, 1],
            linewidth=2.5,
            color=palette[1],
        )
        axs[1, 1].set_title("Validation Structural Similarity Index Measure")
        axs[1, 1].set_xlabel("Epoch")
        axs[1, 1].set_ylabel("SSIM")

        plt.tight_layout(rect=[0, 0.03, 1, 0.94])

        output_path = (
            Path("images")
            / f"training_metrics_{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.png"
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")

        plt.show()
