"""
training code

ResNet paper section 3.4 (Implementation) hyperparams

todo
- saving results
- saving weights
"""
import os
from collections import defaultdict
from time import perf_counter

from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms

from model import VariationalAutoEncoder as VAE, VAEConfig


DATA_DIR = "/Users/williamdreese/percy/vqa/VQA/Images/mscoco/"
# torch.autograd.set_detect_anomaly(True) # use for NaN hunting


class CocoImageDataset(Dataset):
    def __init__(self, image_dir):
        self.image_dir = image_dir
        self.sizes = list()
        self.image_files = sorted(os.listdir(image_dir))

        self.transform = transforms.Compose([
            transforms.Resize(256),  # standard resize
            transforms.RandomResizedCrop(224),  # simple augmentations
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),  # norm
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(img_path).convert("RGB")
        self.sizes.append(image.size)
        image = self.transform(image)
        return image


def log_params(model):
    total_params, total_bytes = 0, 0
    for name, param in model.named_parameters():
        total_params += param.numel()
        total_bytes += param.nelement() * param.element_size()
    param_size_mb = total_bytes / (1024**2)
    print(f"Count: {total_params:,}")
    print(f"Size (Bytes): {total_bytes}")
    print(f"Size (MB): {param_size_mb:.4f}\n")


def kl_divergence(mu, lv):
    return -0.5 * torch.sum(1 + lv - mu.pow(2) - lv.exp(), dim=1)

def nan_check(tensor):
    return torch.isnan(tensor).any().item()

def weight_nan_check(model):
    for name, param in model.named_parameters():
        if nan_check(param.data):
            return True
    return False

def check_explosive_gradients(model):
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            max_grad = param.grad.detach().abs().max()
            if max_grad >= 1e4:
                print(f"Layer: {name} | Max Gradient: {max_grad.item():.4e} <--- Potential Explosion")


if __name__=="__main__":
    # hyperparams
    epochs = 200  # by the time my kids have kids
    global_step = 0
    kl_warmup_steps = 5000
    freeze_step = 4
    unfreeze_step = 40
    beta_step = 400
    frozen = False

    # get some images from the train set
    dset = "train2014"
    dataset = CocoImageDataset(DATA_DIR + f"{dset}/{dset}/")
    loader = DataLoader(dataset, batch_size=128, shuffle=True)

    config = VAEConfig() 
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vae = VAE(config).to(device)
    opt = torch.optim.Adam(vae.parameters(), lr=0.001, weight_decay=0.0001)

    log_params(vae)

    def kl_weighting_base(step, warmup_steps):
        kl_weight = min(0.009, (step-400) / warmup_steps)
        return max(0.003, kl_weight)
    kl_weighting = kl_weighting_base

    for epoch in range(epochs):
        # train mode wrt freezing
        vae.train()  
        if frozen:
            vae._decoder.eval()
        epoch_start = perf_counter()
        for images in loader:
            # apply any phase switches
            if global_step == 0:
                # startup phase - train with a little KL 
                def kl_w1(step, warmup_steps):
                    return 0.01
                kl_weighting = kl_w1
            elif global_step == freeze_step:
                # freeze phase - train encoder only, no KL
                frozen = True
                vae._decoder.requires_grad_(False)
                vae._decoder.eval()  # freeze decoder batchnorms
                opt = torch.optim.Adam(
                    filter(lambda p: p.requires_grad, vae.parameters()),
                    lr=0.001, 
                    weight_decay=0.0001,
                )
                # alter weighting scheme
                def kl_w2(step, warmup_steps):
                    return 0.0
                kl_weighting = kl_w2
            elif global_step == unfreeze_step:
                # unfreeze phase - unfreeze decoder BUT leave beta = 0
                frozen = False
                vae._decoder.requires_grad_(True)
                vae._decoder.train()
                opt = torch.optim.Adam(
                    vae.parameters(),
                    lr=0.001, 
                    weight_decay=0.0001,
                )
                def kl_w3(step, warmup_steps):
                    return 0.003
                kl_weighting = kl_w3
            elif global_step == beta_step:
                # begin re-introducing real beta
                kl_weighting = kl_weighting_base

            step_start = perf_counter()
            images = images.to(device)
            recon, mu, lv = vae(images)
            
            # simple normalized losses
            recon_loss = F.mse_loss(recon, images, reduction="mean")  # avg losses
            kl_loss = kl_divergence(mu, lv)
            kl_weight = kl_weighting(global_step, kl_warmup_steps)
            loss = (recon_loss + kl_weight * kl_loss).mean()

            opt.zero_grad()
            loss.backward()
            # check_explosive_gradients(vae)
            opt.step()

            global_step += 1
            print(f"Step: {global_step}, Loss: {loss} (RL: {recon_loss.mean()}, KL: {kl_loss.mean()}, KLw: {kl_weight})")
            print(f"perf: {perf_counter()-step_start}")
            print(f"mu.mean: {mu.abs().mean().item()}, lv.mean: {lv.mean().item()}")

        print(f"\nEpoch {epoch:03d} | ")
        print(f"Most recent recon: {recon_loss.mean():.1f} | ")
        print(f"KL: {kl_loss.mean():.1f} | ")
        print(f"KL_w: {kl_weight:.3f}\n")
        print(f"Perf: {perf_counter()-epoch_start}")
