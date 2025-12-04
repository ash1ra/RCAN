from utils import create_logger
from typing import Literal

CHANNELS_COUNT = 64
KERNEL_SIZE = 3
REDUCTION = 16
RESIDUAL_GROUPS_COUNT = 10
RESIDUAL_CHANNEL_ATTENTION_BLOCKS_COUNT = 20

SCALING_FACTOR: Literal[2, 4, 8] = 4


logger = create_logger(log_level="INFO", log_file_name="RCAN")
