import os
import glob
import warnings
from argparse import ArgumentParser

import numpy as np
import pyiqa
import torch
from PIL import Image
from tqdm import tqdm
from torchvision import transforms

warnings.filterwarnings("ignore", category=FutureWarning)

parser = ArgumentParser()
parser.add_argument("gt_dir", type=str, help="GT/HR directory")
parser.add_argument("sr_dir", type=str, help="SR directory")

parser.add_argument("--lpips_weight", type=str, default="")
parser.add_argument("--dists_weight", type=str, default="")
parser.add_argument("--niqe_weight", type=str, default="")
parser.add_argument("--musiq_weight", type=str, default="")

parser.add_argument("--skip_lpips", action="store_true")
parser.add_argument("--skip_dists", action="store_true")
parser.add_argument("--skip_fid", action="store_true")
parser.add_argument("--skip_niqe", action="store_true")
parser.add_argument("--skip_musiq", action="store_true")
parser.add_argument("--skip_clipiqa", action="store_true")

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def collect_image_paths(folder):
    exts = ["png", "jpg", "jpeg", "bmp", "tif", "tiff"]
    paths = []
    for ext in exts:
        paths.extend(glob.glob(os.path.join(folder, f"*.{ext}")))
        paths.extend(glob.glob(os.path.join(folder, f"*.{ext.upper()}")))
    return sorted(paths)


psnr = pyiqa.create_metric("psnr", test_y_channel=True, color_space="ycbcr", device=device)
ssim = pyiqa.create_metric("ssim", test_y_channel=True, color_space="ycbcr", device=device)

lpips = None
dists = None
niqe = None
musiq = None
clipiqa = None
fid = None

if not args.skip_lpips:
    lpips_kwargs = {"device": device}
    if args.lpips_weight:
        lpips_kwargs["pretrained_model_path"] = args.lpips_weight
    lpips = pyiqa.create_metric("lpips", **lpips_kwargs)

if not args.skip_dists:
    dists_kwargs = {"device": device}
    if args.dists_weight:
        dists_kwargs["pretrained_model_path"] = args.dists_weight
    dists = pyiqa.create_metric("dists", **dists_kwargs)

if not args.skip_niqe:
    niqe_kwargs = {"device": device}
    if args.niqe_weight:
        niqe_kwargs["pretrained_model_path"] = args.niqe_weight
    niqe = pyiqa.create_metric("niqe", **niqe_kwargs)

if not args.skip_musiq:
    musiq_kwargs = {"device": device}
    if args.musiq_weight:
        musiq_kwargs["pretrained_model_path"] = args.musiq_weight
    musiq = pyiqa.create_metric("musiq", **musiq_kwargs)

if not args.skip_clipiqa:
    clipiqa = pyiqa.create_metric("clipiqa", device=device)

if not args.skip_fid:
    fid = pyiqa.create_metric("fid", device=device)

test_sr_paths = collect_image_paths(args.sr_dir)
test_hr_paths = collect_image_paths(args.gt_dir)

print("GT dir:", args.gt_dir)
print("SR dir:", args.sr_dir)
print("Found GT:", len(test_hr_paths))
print("Found SR:", len(test_sr_paths))

assert len(test_sr_paths) == len(test_hr_paths), \
    f"SR count ({len(test_sr_paths)}) != HR count ({len(test_hr_paths)})"

metrics = {
    "psnr": [],
    "ssim": [],
}

if lpips is not None:
    metrics["lpips"] = []
if dists is not None:
    metrics["dists"] = []
if niqe is not None:
    metrics["niqe"] = []
if musiq is not None:
    metrics["musiq"] = []
if clipiqa is not None:
    metrics["clipiqa"] = []

for sr_path, hr_path in tqdm(list(zip(test_sr_paths, test_hr_paths)), total=len(test_sr_paths)):
    sr = Image.open(sr_path).convert("RGB")
    sr = transforms.ToTensor()(sr).to(device).unsqueeze(0)

    hr = Image.open(hr_path).convert("RGB")
    hr = transforms.ToTensor()(hr).to(device).unsqueeze(0)

    metrics["psnr"].append(psnr(sr, hr).item())
    metrics["ssim"].append(ssim(sr, hr).item())

    if lpips is not None:
        metrics["lpips"].append(lpips(sr, hr).item())
    if dists is not None:
        metrics["dists"].append(dists(sr, hr).item())
    if niqe is not None:
        metrics["niqe"].append(niqe(sr).item())
    if musiq is not None:
        metrics["musiq"].append(musiq(sr).item())
    if clipiqa is not None:
        metrics["clipiqa"].append(clipiqa(sr).item())

for k in metrics:
    metrics[k] = float(np.mean(metrics[k]))

if fid is not None:
    metrics["fid"] = float(fid(args.sr_dir, args.gt_dir))

for k, v in metrics.items():
    if k == "niqe":
        print(k, f"{v:.3g}")
    elif k == "fid":
        print(k, f"{v:.5g}")
    else:
        print(k, f"{v:.4g}")