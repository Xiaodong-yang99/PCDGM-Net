# PCDGM-Net

This repository provides the implementation, pretrained model weights, evaluation weights, and reproduction instructions for PCDGM-Net.

## Environment

The experiments were conducted with the following environment:

- Python 3.10.8
- CUDA 12.1
- PyTorch 2.1.2
- Torchvision 0.16.2
- MMCV 2.2.0
- MMEngine 0.10.7
- Mamba-SSM 2.3.1

Other Python dependencies are listed in `requirements.txt`.

Install the remaining dependencies with:

```bash
pip install -r requirements.txt
```

## Training

Before training, open `config.yml` and modify the first line to specify the path to your own training dataset.

For example:

```yaml
trainset: /path/to/your/training_dataset
```

Then run:

```bash
python train.py
```

## Testing

The test script accepts the low-resolution input directory, output directory, and model directory as command-line arguments. Each directory should contain the images directly (the script does not search recursively). For the NPU_DJI100_1 subset, run:

```bash
python test.py \
  --LR_dir ./testests_lr_x4/NPU_DJI100_1 \
  --SR_dir ./SR_results/NPU_DJI100_1 \
  --model_dir ./weight/PCDGM_Net
```

The script loads `latest.pkl` from the directory supplied to `--model_dir` and writes the reconstructed images to `--SR_dir`.

## Evaluation

The reconstructed SR images can be evaluated using `evaluate.py`. Its first positional argument is the HR/GT image directory and its second is the reconstructed SR image directory. The script reads image files directly from these directories and does not search recursively, so use the directory that contains the images rather than a dataset root. Ensure the two directories contain the same images in matching sorted filename order. For NPU_DJI100_1, run:

```bash
python evaluate.py \
  ./testests/NPU_DJI100_1/NPU_DJI100_1 \
  ./SR_results/NPU_DJI100_1 \
  --lpips_weight ./weights/LPIPS_v0.1_alex-df73285e.pth \
  --dists_weight ./weights/DISTS_weights-f5e65c96.pth \
  --niqe_weight ./weights/niqe_modelparameters.mat \
  --musiq_weight ./weights/musiq_spaq_ckpt-358bb6af.pth
```

## Pretrained Weights

The pretrained PCDGM-Net checkpoint is tracked at `weight/PCDGM_Net/latest.pkl`. The evaluation metric weights are tracked under `weights/`. The `.pkl` and `.pth` files use Git LFS; install Git LFS and run `git lfs pull` after cloning to retrieve the weight files before testing or evaluation.

## Acknowledgements

This work is developed based on LDFF-Net. We sincerely thank the authors of LDFF-Net for sharing their code and making their implementation publicly available.
