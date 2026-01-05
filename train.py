"""
training code

- ResNet paper section 3.4 (Implementation) hyperparams

results saved in 
- /logs/<run_id>/logfile.txt
- /logs/<run_id>/step_N.jpg

todo
- saving weights
"""
import os
import sys
from collections import defaultdict
from time import perf_counter

from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms

from model import VariationalAutoEncoder as VAE, VAEConfig


DATA_DIR = "/Users/williamdreese/percy/vqa/VQA/Images/mscoco/"
# torch.autograd.set_detect_anomaly(True) # use for NaN hunting


class CocoImageDataset(Dataset):
    def __init__(self, image_dir, count=None):
        self.image_dir = image_dir
        self.sizes = list()
        self.image_files = sorted(os.listdir(image_dir))
        if count:
            self.image_files = self.image_files[:count]

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


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

@torch.no_grad()
def save_x_and_recon(x, x_hat, idx=0, filename=None,
                     mean=IMAGENET_MEAN, std=IMAGENET_STD):
    """
    x, x_hat: [B,3,H,W] tensors in normalized space (e.g., ImageNet Normalize).
    idx: which element in the batch to show
    """
    assert x.dim() == 4 and x.shape[1] == 3
    assert x_hat.shape == x.shape
    assert 0 <= idx < x.shape[0]

    # pick one example
    x0 = x[idx].detach().cpu().float()
    xh0 = x_hat[idx].detach().cpu().float()

    # CHW -> HWC
    x0  = x0.permute(1, 2, 0)
    xh0 = xh0.permute(1, 2, 0)

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    if filename:
        fig.suptitle(filename, fontsize=14)

    axes[0].imshow(x0)
    axes[0].set_title("input")
    axes[0].axis("off")

    axes[1].imshow(xh0)
    axes[1].set_title("recon")
    axes[1].axis("off")

    plt.tight_layout()
    # plt.show()
    plt.savefig(filename)


def check_explosive_gradients(model):
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            max_grad = param.grad.detach().abs().max()
            if max_grad >= 1e4:
                print(f"Layer: {name} | Max Gradient: {max_grad.item():.4e} <--- Potential Explosion")

def image_stats(name, t):
    t = t.detach()
    print(
        f"{name}: shape={tuple(t.shape)} dtype={t.dtype} "
        f"min={t.min().item():.4f} max={t.max().item():.4f} mean={t.mean().item():.4f}"
    )

if __name__=="__main__":
    run_id = sys.argv[1] if len(sys.argv) > 1 else "default"

    # hyperparams
    epochs = 20_000  # by the time my kids have kids
    global_step = 0
    kl_warmup_steps = 5000
    freeze_step = 4
    unfreeze_step = 40
    beta_step = 400
    frozen = False

    # get some images from the train set
    dset = "train2014"
    dataset = CocoImageDataset(DATA_DIR + f"{dset}/{dset}/", count=None)
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
            print(f"\nStep: {global_step}, Loss: {loss} (RL: {recon_loss.mean()}, KL: {kl_loss.mean()}, KLw: {kl_weight})")
            print(f"perf: {perf_counter()-step_start}")
            print(f"mu.mean: {mu.abs().mean().item()}, lv.mean: {lv.mean().item()}")
            print(f"mu.pdist: {torch.pdist(mu).mean().item()}")

            # is my latent space even being used
            '''
            with torch.no_grad():
                z_prior = torch.randn_like(mu)
                x_prior = vae._decoder(z_prior)
                prior_recon_loss = F.mse_loss(x_prior, images, reduction="mean")
                print(f"prior MSE: {prior_recon_loss}")
            '''

            # visualize
            if global_step % 50 == 1:
                #image_stats("x", images)
                #image_stats("x_hat", recon)
                @torch.no_grad()
                def denorm_imagenet(t):
                    mean = torch.tensor(IMAGENET_MEAN, device=t.device).view(1,3,1,1)
                    std  = torch.tensor(IMAGENET_STD,  device=t.device).view(1,3,1,1)
                    return (t * std + mean)
                image_display = denorm_imagenet(images).clamp(0, 1)
                recon_display = denorm_imagenet(recon).clamp(0, 1)
                #image_stats("x_norm", image_display)
                #image_stats("x_hat_norm", recon_display)
                save_x_and_recon(image_display, recon_display, filename=run_id+f"/step_{global_step}.png")

        '''
        print(f"\nEpoch {epoch:03d} | ")
        print(f"Most recent recon: {recon_loss.mean():.1f} | ")
        print(f"KL: {kl_loss.mean():.1f} | ")
        print(f"KL_w: {kl_weight:.3f}\n")
        print(f"Perf: {perf_counter()-epoch_start}")
        '''
