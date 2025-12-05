import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

import config
from dataset import SRDataset
from models import ResidualChannelAttentionNetwork
from trainer import Trainer


def main() -> None:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_dataset = SRDataset(
        data_path=config.TRAIN_DATASET_PATH,
        scaling_factor=config.SCALING_FACTOR,
        crop_size=config.CROP_SIZE,
        dev_mode=config.DEV_MODE,
    )

    val_dataset = SRDataset(
        data_path=config.VALIDATION_DATASET_PATH,
        scaling_factor=config.SCALING_FACTOR,
        crop_size=config.CROP_SIZE,
        dev_mode=config.DEV_MODE,
    )

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=config.TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True if device == "cuda" else False,
        prefetch_factor=config.PREFETCH_FACTOR,
        persistent_workers=True,
    )

    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=config.TEST_BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True if device == "cuda" else False,
        prefetch_factor=config.PREFETCH_FACTOR,
        persistent_workers=True,
    )

    model = ResidualChannelAttentionNetwork(
        channels_count=config.CHANNELS_COUNT,
        kernel_size=config.KERNEL_SIZE,
        reduction=config.REDUCTION,
        rg_count=config.RESIDUAL_GROUPS_COUNT,
        rcab_count=config.RESIDUAL_CHANNEL_ATTENTION_BLOCKS_COUNT,
        scaling_factor=config.SCALING_FACTOR,
    )

    loss_fn = nn.L1Loss()

    optimizer = Adam(
        params=model.parameters(),
        lr=config.LEARNING_RATE,
        betas=(config.ADAM_BETA1_VALUE, config.ADAM_BETA2_VALUE),
        eps=config.ADAM_EPSILON,
    )

    scheduler = StepLR(
        optimizer=optimizer,
        step_size=config.SCHEDULER_STEP_SIZE,
        gamma=config.SCHEDULER_GAMMA,
    )

    trainer = Trainer(
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        epochs=config.EPOCHS,
        scheduler=scheduler,
        device=device,
    )

    if (
        config.LOAD_BEST_RCAN_CHECKPOINT
        and config.BEST_RCAN_CHECKPOINT_DIR_PATH.exists()
    ):
        trainer.load_checkpoint(config.BEST_RCAN_CHECKPOINT_DIR_PATH)
    elif config.LOAD_RCAN_CHECKPOINT and config.LOAD_RCAN_CHECKPOINT_DIR_PATH.exists():
        trainer.load_checkpoint(config.LOAD_RCAN_CHECKPOINT_DIR_PATH)

    trainer.train()


if __name__ == "__main__":
    main()
