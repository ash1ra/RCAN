# RCAN for SISR task using PyTorch

This project implements an **RCAN** (Residual Channel Attention Network) model for the **SISR** (Single Image Super-Resolution) task. The primary goal is to upscale low-resolution (LR) images by a given factor (2x, 4x, 8x) to produce super-resolution (SR) images with high fidelity and perceptual quality.

This implementation is based on the paper [Image Super-Resolution Using Very Deep Residual Channel Attention Networks](https://arxiv.org/pdf/1807.02758).

***Note:** I have not been able to achieve good quality using this implementation for reasons I am not fully aware of yet. I used the parameters specified in the study but trained the model for a shorter amount of time. Detailed analysis is provided below.*

## Demonstration

The following images compare standard bicubic interpolation with the output of the RCAN model.

![Baboon comparison image](images/comparison_img_baboon.png)
![Butterfly comparison image](images/comparison_img_butterfly.png)
![Bird comparison image](images/comparison_img_bird.png)
![Man comparison image](images/comparison_img_man.png)
![PPT3 comparison image](images/comparison_img_ppt3.png)

## Key Features

- **Residual in Residual (RIR)** structure enables the training of extremely deep networks by nesting residual groups to bypass abundant low-frequency information and focus purely on learning high-frequency details.
- **Channel Attention (CA)** mechanism distinguishes this model from standard ResNets by using a "Squeeze-and-Excitation" approach that adaptively rescales channel-wise features to prioritize key information.
- **Scalable and Modular Design** allows for easy adjustment of network depth and width via `config.py` through the use of modular `ResidualGroup` and `RCAB` blocks.

## Datasets

### Training

The model is trained on the **DIV2K** dataset. The `SRDataset` class in `dataset.py` dynamically creates LR images from HR images using bicubic downsampling and applies random crops and augmentations (flips, rotations). HR images are patches extracted from the original DIV2K images and were created using the `prepare_dataset.py` script with `patch_size=480` and `stride=240`.

### Validation

The **DIV2K_valid** dataset is used for validation.

### Testing

The `test.py` script is configured to evaluate the trained model on standard benchmark datasets: **Set5**, **Set14**, **BSDS100**, **Urban100**, and **Manga109**.

## Project Structure

```
.
├── checkpoints/             # Model weights (.safetensors) and training states
├── images/                  # Inference inputs, outputs, and training plots
├── config.py                # Hyperparameters and file paths
├── dataset.py               # SRDataset class and image transformations
├── inference.py             # Inference pipeline
├── models.py                # RCAN model architecture definition
├── test.py                  # Testing pipeline
├── trainer.py               # Trainer class for model training
├── train.py                 # Training pipeline
└── utils.py                 # Utility functions
```

## Configuration

All hyperparameters, paths, and training settings can be configured in the `config.py` file.

Explanation of some settings:
- `LOAD_RCAN_CHECKPOINT`: Set to `True` to resume training from the specified RCAN checkpoint (for `train.py`).
- `LOAD_BEST_RCAN_CHECKPOINT`: Set to `True` to resume training from the best RCAN checkpoint (for `train.py`).
- `TRAIN_DATASET_PATH`: Path to the training data. Can be a directory of images or a `.txt` file listing image paths.
- `VAL_DATASET_PATH`: Path to the validation data. Can be a directory of images or a `.txt` file listing image paths.
- `TEST_DATASETS_PATHS`: List of paths to the test data. Each path can be a directory of images or a `.txt` file listing image paths.
- `DEV_MODE`: Set to `True` to use a 10% subset of the training data for quick testing.

## Setting Up and Running the Project

### 1. Installation

1. Clone the repository:
```bash
git clone https://github.com/ash1ra/RCAN.git
cd RCAN
```

2. Create a `.venv` and install dependencies:
```bash
uv sync
```

3. Activate the virtual environment:
```bash
# On Windows
.venv\Scripts\activate
# On Unix or MacOS
source .venv/bin/activate
```

### 2. Data Preparation

1.  [Download](https://data.vision.ee.ethz.ch/cvl/DIV2K/) the **DIV2K** datasets (`Train Data (HR images)` and `Validation Data (HR images)`).
2.  [Download](https://figshare.com/articles/dataset/BSD100_Set5_Set14_Urban100/21586188) the standard benchmark datasets (**Set5**, **Set14**, **BSDS100**, **Urban100**) and [Manga109](http://www.manga109.org/en/download.html) dataset.
3.  Organize your data directory as expected by the scripts:
    ```
    data/
    ├── DIV2K/
    │   ├── 1.jpg
    │   └── ...
    ├── DIV2K_valid/
    │   ├── 1.jpg
    │   └── ...
    ├── Set5/
    │   ├── baboon.png
    │   └── ...
    ├── Set14/
    │   └── ...
    ...
    ```

    or
    
    ```
    data/
    ├── DIV2K.txt
    ├── DIV2K_valid.txt
    ├── Set5.txt
    ├── Set14.txt
    ...
    ```
    
4.  Update the paths (`TRAIN_DATASET_PATH`, `VAL_DATASET_PATH`, `TEST_DATASETS_PATHS`) in `config.py` to match your data structure.

### 3. Training

1.  Adjust parameters in `config.py` as needed.
2.  Run the training script:
    ```bash
    python train.py
    ```
3.  Training progress will be logged to the console and to a file in the `logs/` directory.
4.  Checkpoints will be saved in `checkpoints/`. A plot of the training metrics will be saved in `images/` upon completion.

### 4. Testing

To evaluate the model's performance on the test datasets:

1.  Ensure the `BEST_RCAN_CHECKPOINT_DIR_PATH` in `config.py` points to your trained model (e.g., `checkpoints/rcan_best`).
2.  Run the test script:
    ```bash
    python test.py
    ```
3.  The script will print the average PSNR and SSIM for each dataset.

### 5. Inference

To upscale a single image:

1.  Place your image in the `images/` folder (or update the path).
2.  In `config.py`, set `INFERENCE_INPUT_IMG_PATH` to your image, `INFERENCE_OUTPUT_IMG_PATH` to desired location of output image, `INFERENCE_COMPARISON_IMG_PATH` to deisred location of comparison image (optional) and `BEST_RCAN_CHECKPOINT_DIR_PATH` to your trained model.
3.  Run the script:
    ```bash
    python inference.py
    ```
4.  The upscaled image (`sr_img_*.png`) and a comparison image (`comparison_img_*.png`) will be saved in the `images/` directory.

## Training Results

![RCAN model training metrics](images/rcan_training_metrics.png)

The model was trained for 200 epochs with a batch size of 16 on an NVIDIA RTX 4060 Ti (8 GB), which took nearly 24 hours. The training dataset consisted of 31802 patches obtained by cropping 800 images from the DIV2K dataset into 480px patches with a 240px stride. The rest of the hyperparameters are specified in the chart. The final model selected is the one with the highest PSNR on the validation set.

## Benchmark Evaluation (4x Upscaling)

The final model (`rcan_best`) was evaluated on standard benchmark datasets. Metrics are calculated on the Y-channel after shaving 4px (the scaling factor) from the border.

**PSNR (dB) / SSIM Comparison**
| Dataset | RCAN (this project) | RCAN (paper)
| :--- | :---: | :---:
| **Set5** | 32.43/0.8967 | 32.63/0.9002
| **Set14** | 27.47/0.7873 | 28.87/0.7889
| **BSDS100** | 26.23/0.7408 | 27.77/0.7436
| **Urban100**| 24.92/0.7996 | 26.82/0.8087
| **Manga109** | 29.31/0.9142 | 31.22/0.9173

***Note**: There are multiple factors that can lead to differences in benchmark results: the original paper uses MATLAB functions to evaluate image quality, whereas I use PyTorch functions; I trained the model for significantly less time than in the original paper, with the learning rate decayed 5 times more frequently; and other minor factors such as different image cropping techniques or LR image generation methods.*

## Visual Comparisons

The following images compare the standard bicubic interpolation with the output of the RCAN model. I selected various images where the difference in results would be visible, including anime images, photos, etc.

![Comparisson image 1](images/comparison_img_1.png)
![Comparisson image 2](images/comparison_img_2.png)
![Comparisson image 3](images/comparison_img_3.png)
![Comparisson image 4](images/comparison_img_4.png)
![Comparisson image 5](images/comparison_img_5.png)

## Acknowledgements

This implementation is based on the paper [Image Super-Resolution Using Very Deep Residual Channel Attention Networks](https://arxiv.org/pdf/1807.02758).

```bibtex
@misc{zhang2018imagesuperresolutionusingdeep,
      title={Image Super-Resolution Using Very Deep Residual Channel Attention Networks}, 
      author={Yulun Zhang and Kunpeng Li and Kai Li and Lichen Wang and Bineng Zhong and Yun Fu},
      year={2018},
      eprint={1807.02758},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/1807.02758}, 
}
```

DIV2K dataset citation:

```bibtex
@InProceedings{Timofte_2018_CVPR_Workshops,
  author = {Timofte, Radu and Gu, Shuhang and Wu, Jiqing and Van Gool, Luc and Zhang, Lei and Yang, Ming-Hsuan and Haris, Muhammad and others},
  title = {NTIRE 2018 Challenge on Single Image Super-Resolution: Methods and Results},
  booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
  month = {June},
  year = {2018}
}
```

Manga109 dataset citation:
```bibtex
@article{mtap_matsui_2017,
    author={Yusuke Matsui and Kota Ito and Yuji Aramaki and Azuma Fujimoto and Toru Ogawa and Toshihiko Yamasaki and Kiyoharu Aizawa},
    title={Sketch-based Manga Retrieval using Manga109 Dataset},
    journal={Multimedia Tools and Applications},
    volume={76},
    number={20},
    pages={21811--21838},
    doi={10.1007/s11042-016-4020-z},
    year={2017}
}

@article{multimedia_aizawa_2020,
    author={Kiyoharu Aizawa and Azuma Fujimoto and Atsushi Otsubo and Toru Ogawa and Yusuke Matsui and Koki Tsubota and Hikaru Ikuta},
    title={Building a Manga Dataset ``Manga109'' with Annotations for Multimedia Applications},
    journal={IEEE MultiMedia},
    volume={27},
    number={2},
    pages={8--18},
    doi={10.1109/mmul.2020.2987895},
    year={2020}
}
```

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

