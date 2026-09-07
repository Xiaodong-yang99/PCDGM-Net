import torch, os, random
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.backends import cudnn
import numpy as np
from argparse import ArgumentParser
import time
from tqdm import tqdm
from omegaconf import OmegaConf

from dataset import RealESRGANDataset, RealESRGANDegrader
from loss_utils import l1_loss, wavelet_loss


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main(mymodel, args, config):
    epoch = args.epoch
    learning_rate = args.learning_rate
    bsz = args.batch_size

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True

    dataset = RealESRGANDataset(config, bsz)
    degrader = RealESRGANDegrader(config, device)

    dataloader = DataLoader(
        dataset,
        batch_size=bsz,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )

    optimizer = torch.optim.Adam(mymodel.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=args.milestones,
        gamma=0.5
    )

    model_dir = args.model_dir
    log_dir = args.log_dir
    log_path = os.path.join(log_dir, f"{args.log_name}.txt")

    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    best_loss = float("inf")

    print("start training...")
    for epoch_i in range(1, epoch + 1):
        start_time = time.time()
        mymodel.train()

        loss_avg = 0.0
        loss_L1 = 0.0
        loss_WAV = 0.0
        iter_num = 0

        for batch in tqdm(dataloader):
            with torch.cuda.amp.autocast(enabled=False):
                LR, HR = degrader.degrade(batch)
                LR, HR = LR * 2 - 1, HR * 2 - 1
                LR = LR.to(device)
                HR = HR.to(device)
                optimizer.zero_grad()
                SR = mymodel(LR)

                loss_l1 = l1_loss(HR, SR)
                loss_wav = wavelet_loss(HR, SR)
                loss =  loss_l1 + 0.5*loss_wav

                loss_avg+=loss

                loss.backward()
                optimizer.step()
            iter_num+=1

        scheduler.step()

        loss_avg /= iter_num
        loss_L1 /= iter_num
        loss_WAV /= iter_num

        cur_lr = scheduler.get_last_lr()[0]
        log_data = (
            f"[{epoch_i}/{epoch}] "
            f"Average loss: {loss_avg:.6f}, "
            f"loss_L1: {loss_L1:.6f}, "
            f"loss_WAV: {loss_WAV:.6f}, "
            f"time cost: {time.time() - start_time:.2f}s, "
            f"cur lr: {cur_lr:.8f}"
        )

        print(log_data)
        with open(log_path, "a", encoding="utf-8") as log_file:
            log_file.write(log_data + "\n")

        # 保存 latest
        torch.save(
            {
                "epoch": epoch_i,
                "model_state_dict": mymodel.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
            },
            os.path.join(model_dir, "latest.pkl")
        )

        # 保存 best
        if loss_avg < best_loss:
            best_loss = loss_avg
            torch.save(
                {
                    "epoch": epoch_i,
                    "model_state_dict": mymodel.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_loss": best_loss,
                },
                os.path.join(model_dir, "best_loss.pkl")
            )

        # 周期保存
        if epoch_i % args.save_interval == 0:
            torch.save(
                {
                    "epoch": epoch_i,
                    "model_state_dict": mymodel.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                },
                os.path.join(model_dir, f"net_params_{epoch_i}.pkl")
            )


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cudnn.benchmark = True
    set_seed(42)

    # 给新模型单独命名，自动保存到新文件夹
    model_name = "LDFF_Net_HDGM_PDRF"

    from model.LDFF_Net_HDGM_PDRF import MYMODEL
    mymodel = MYMODEL(up_scale=4).to(device)

    config = OmegaConf.load("config.yml")

    parser = ArgumentParser()
    parser.add_argument("--epoch", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--milestones", type=int, nargs='+', default=[200, 400, 600, 800])
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--model_dir", type=str, default=f"weight/{model_name}")
    parser.add_argument("--log_dir", type=str, default="log")
    parser.add_argument("--log_name", type=str, default=model_name)
    parser.add_argument("--save_interval", type=int, default=200)

    args = parser.parse_args()
    print(f"开始处理: {model_name}")
    main(mymodel, args, config)