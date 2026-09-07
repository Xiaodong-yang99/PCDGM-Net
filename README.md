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

Before testing, open `test.py` and modify the following paths according to your local environment:

```python
LR_path = '/path/to/your/LR_images'
SR_path = '/path/to/save/SR_results'
```

Then run:

```bash
python test.py
```

The pretrained model weights are loaded according to the weight path specified in `test.py`.

## Evaluation

The reconstructed SR images can be evaluated using `evaluate.py`. The first positional argument specifies the HR/GT image directory, while the second positional argument specifies the reconstructed SR image directory.

Run:

```bash
python evaluate.py \
/path/testests \
/path/SR \
--lpips_weight /root/LDFF-Net/weights/LPIPS_v0.1_alex-df73285e.pth \
--dists_weight /root/LDFF-Net/weights/DISTS_weights-f5e65c96.pth \
--niqe_weight /root/LDFF-Net/weights/niqe_modelparameters.mat \
--musiq_weight /root/LDFF-Net/weights/musiq_spaq_ckpt-358bb6af.pth
```

## Pretrained Weights

The pretrained PCDGM-Net model weight is provided in the `weight/` directory.
