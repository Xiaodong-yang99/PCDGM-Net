# ************************************************************************************ #
import torch, os, glob, copy
import torch.nn.functional as F
import numpy as np
from PIL import Image
from argparse import ArgumentParser
from torchvision import transforms
from tqdm import tqdm
from torch.backends import cudnn


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 首先设置GPU设备
cudnn.benchmark = True
model_name = "PCDGM_Net"
from model.PCDGM_Net import MYMODEL
mymodel = MYMODEL(up_scale=4)
mymodel = mymodel.to(device)
LR_path = 'your LR path's
SR_path = 'your SR path'
device = torch.device("cuda")
parser = ArgumentParser()
parser.add_argument("--epoch", type=int, default=1000)
parser.add_argument("--model_dir", type=str, default=f"weight/{model_name}")
parser.add_argument("--LR_dir", type=str, default=LR_path)
parser.add_argument("--SR_dir", type=str, default=SR_path)
args = parser.parse_args()


# mymodel.load_state_dict(torch.load("./%s/net_params_%d.pkl" % (args.model_dir, args.epoch), weights_only=False))
weight_path = os.path.join(args.model_dir, "latest.pkl")

# 读取权重
checkpoint = torch.load(weight_path, map_location=device)

# 如果保存时是训练字典
if "model_state_dict" in checkpoint:
    mymodel.load_state_dict(checkpoint["model_state_dict"])
else:
    mymodel.load_state_dict(checkpoint)
    
mymodel = mymodel.to(device)
mymodel.eval()

# test_LR_paths = list(sorted(glob.glob(os.path.join(args.LR_dir, "*.png"))))
test_LR_paths = []
for ext in ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff"]:
    test_LR_paths.extend(glob.glob(os.path.join(args.LR_dir, ext)))

test_LR_paths = list(sorted(test_LR_paths))

print("LR_dir:", args.LR_dir)
print("Found LR images:", len(test_LR_paths))
print("First 5 LR images:", test_LR_paths[:5])

if len(test_LR_paths) == 0:
    raise RuntimeError("No LR images found. Please check LR_dir path and image extensions.")
# test_HR_paths = list(sorted(glob.glob(os.path.join(args.HR_dir, "*.png"))))

os.makedirs(args.SR_dir, exist_ok=True)

total_time = 0
with torch.no_grad():
    dummy_input = torch.rand(1, 3, 224, 224).cuda()
    for _ in range(10):
        _ = mymodel(dummy_input)


start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

with torch.no_grad():
    mypad=2
    for i, path in enumerate(tqdm(test_LR_paths)):
        LR_img = Image.open(path).convert("RGB")
        H, W = LR_img.size

        pad_h = (mypad - H % mypad) % mypad
        pad_w = (mypad - W % mypad) % mypad

        if pad_h > 0 or pad_w > 0:

            lr_np = np.array(LR_img)

            lr_padded = np.pad(lr_np, ((0, pad_w), (0, pad_h), (0, 0)), mode='edge')
            LR_img = Image.fromarray(lr_padded)

        LR = transforms.ToTensor()(LR_img).to(device).unsqueeze(0) * 2 - 1

        torch.cuda.synchronize()
        start_event.record()

        SR = mymodel(LR)

        end_event.record()
        torch.cuda.synchronize()

        total_time += start_event.elapsed_time(end_event)

        SR = (SR - SR.mean(dim=[2, 3], keepdim=True)) / SR.std(dim=[2, 3], keepdim=True) \
             * LR.std(dim=[2, 3], keepdim=True) + LR.mean(dim=[2, 3], keepdim=True)

        SR_img = transforms.ToPILImage()((SR[0] / 2 + 0.5).clamp(0, 1).cpu())

        if pad_h > 0 or pad_w > 0:
            target_width = W * mypad
            target_height = H * mypad
            SR_img = SR_img.crop((0, 0, target_height, target_width))

        SR_img.save(os.path.join(args.SR_dir, os.path.basename(path)))

    print(f"处理 {len(test_LR_paths)} 张图像的平均时间: {total_time/len(test_LR_paths):.2f} 毫秒")