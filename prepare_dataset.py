from pathlib import Path

from PIL import Image
from tqdm import tqdm

import config


def cut_images(
    input_data_path: Path,
    output_data_path: Path,
    patch_size: int,
    stride: int,
) -> None:
    output_data_path.mkdir(parents=True, exist_ok=True)

    if input_data_path.exists():
        if input_data_path.is_dir():
            config.logger.info(
                f'Reading images from directory ("{input_data_path}")...'
            )
            img_paths = list(input_data_path.iterdir())
        elif input_data_path.is_file():
            config.logger.info(f'Reading images from file ("{input_data_path}")...')
            with open(input_data_path, "r") as f:
                img_paths = [Path(line.strip()) for line in f.readlines() if line]
    else:
        config.logger.error(f"Input file {input_data_path} not found.")
        raise FileNotFoundError

    config.logger.info(f"Found {len(img_paths)} images.")

    total_patches = 0
    for img_path in tqdm(img_paths, desc="Processing images"):
        if not img_path.exists():
            continue
        try:
            with Image.open(img_path) as img:
                img = img.convert("RGB")
                img_width, img_height = img.size

                if img_width < patch_size or img_height < patch_size:
                    continue

                patch_idx = 0

                for y in range(0, img_height - patch_size + 1, stride):
                    for x in range(0, img_width - patch_size + 1, stride):
                        patch = img.crop((x, y, x + patch_size, y + patch_size))

                        save_name = f"{img_path.stem}_p{patch_idx:03d}.png"
                        patch.save(
                            output_data_path / save_name, "PNG", compress_level=1
                        )

                        patch_idx += 1
                        total_patches += 1

                if img_width % stride != 0 and img_width > patch_size:
                    for y in range(0, img_height - patch_size + 1, stride):
                        patch = img.crop(
                            (img_width - patch_size, y, img_width, y + patch_size)
                        )
                        save_name = f"{img_path.stem}_p{patch_idx:03d}.png"
                        patch.save(
                            output_data_path / save_name, "PNG", compress_level=1
                        )

                        patch_idx += 1
                        total_patches += 1

                if img_height % stride != 0 and img_height > patch_size:
                    for x in range(0, img_width - patch_size + 1, stride):
                        patch = img.crop(
                            (x, img_height - patch_size, x + patch_size, img_height)
                        )
                        save_name = f"{img_path.stem}_p{patch_idx:03d}.png"
                        patch.save(
                            output_data_path / save_name, "PNG", compress_level=1
                        )

                        patch_idx += 1
                        total_patches += 1
        except Exception as e:
            config.logger.error(e)

    config.logger.info(f"Done! Total patches created: {total_patches}")


if __name__ == "__main__":
    cut_images(
        input_data_path=Path("data/DIV2K_train.txt"),
        output_data_path=Path("data/DIV2K_train_patches"),
        patch_size=480,
        stride=240,
    )
