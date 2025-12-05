from utils import create_logger
from typing import Literal
from pathlib import Path

CHANNELS_COUNT = 64
KERNEL_SIZE = 3
REDUCTION = 16
RESIDUAL_GROUPS_COUNT = 10
RESIDUAL_CHANNEL_ATTENTION_BLOCKS_COUNT = 20

SCALING_FACTOR: Literal[2, 4, 8] = 4
CROP_SIZE = 48
LEARNING_RATE = 1e-4
BATCH_SIZE = 16
TEST_BATCH_SIZE = 1
EPOCHS = 500
PRINT_FREQUENCY = 200

NUM_WORKERS = 8
PREFETCH_FACTOR = 4

SCHEDULER_MILESTONES = []
SCHEDULER_GAMMA = 0.5

ADAM_BETA1_VALUE = 0.9
ADAM_BETA2_VALUE = 0.999
ADAM_EPSILON = 1e-8

DEV_MODE = False

TRAIN_DATASET_PATH = Path("data/DIV2K_train.txt")
VALIDATION_DATASET_PATH = Path("data/DIV2K_val.txt")
TEST_DATASET_PATHS = [
    Path("data/Set5.txt"),
    Path("data/Set14.txt"),
    Path("data/BSDS100.txt"),
    Path("data/Urban100.txt"),
    Path("data/Manga109.txt"),
]

logger = create_logger(log_level="INFO", log_file_name="RCAN")
