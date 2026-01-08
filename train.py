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


### Training, eval, & test loading

class CocoImageDataset(Dataset):
    def __init__(self, image_dir, count=None, rrc=True):
        self.image_dir = image_dir
        self.sizes = list()
        self.image_files = sorted(os.listdir(image_dir))
        if count:
            self.image_files = self.image_files[:count]

        # training vs validation/test
        if rrc:
            trans = [
                transforms.Resize(256), # standard resize
                transforms.RandomResizedCrop(
                    224, 
                    scale=(0.75, 1.0), 
                    ratio=(3/4, 4/3),
                ),  # simple augmentations
            ]
        else:
            trans = [transforms.Resize((224,224))]

        trans.extend([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])
        self.transform = transforms.Compose(trans)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(img_path).convert("RGB")
        self.sizes.append(image.size)
        image = self.transform(image)
        return image

### Logging & visualization

def log_params(model):
    total_params, total_bytes = 0, 0
    for name, param in model.named_parameters():
        total_params += param.numel()
        total_bytes += param.nelement() * param.element_size()
    param_size_mb = total_bytes / (1024**2)
    print(f"Count: {total_params:,}")
    print(f"Size (Bytes): {total_bytes}")
    print(f"Size (MB): {param_size_mb:.4f}")

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
def save_x_and_recon(
    x, 
    x_hat, 
    x_hat_mu, 
    idx=9, 
    filename=None,
    mean=IMAGENET_MEAN, 
    std=IMAGENET_STD,
):
    """
    x, x_hat, x_hat_mu: [B,3,H,W] tensors in normalized space
    idx: which element in the batch to show
    filename: what to name the saveed file for later viewing
    mean, std: input transformation values
    """
    assert x.dim() == 4 and x.shape[1] == 3
    assert x_hat.shape == x.shape 
    assert x.shape == x_hat_mu.shape
    assert 0 <= idx < x.shape[0]

    # pick one example
    x0 = x[idx].detach().cpu().float()
    xh0 = x_hat[idx].detach().cpu().float()
    xhu0 = x_hat_mu[idx].detach().cpu().float()

    # CHW -> HWC
    x0  = x0.permute(1, 2, 0)
    xh0 = xh0.permute(1, 2, 0)
    xhu0 = xhu0.permute(1, 2, 0)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    if filename:
        fig.suptitle(filename, fontsize=14)

    axes[0].imshow(x0)
    axes[0].set_title("input")
    axes[0].axis("off")

    axes[1].imshow(xh0)
    axes[1].set_title("recon")
    axes[1].axis("off")

    axes[2].imshow(xhu0)
    axes[2].set_title("recon_u")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(filename)
    plt.close(fig)  # free up pic memory

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

# save here for when I need to verify latent is being used
def test_mu(mu, vae, images, recon):
    # is my latent space even being used
    with torch.no_grad():
        z_prior = torch.randn_like(mu)
        x_prior = vae._decoder(z_prior)
        prior_recon_loss = F.mse_loss(x_prior, images, reduction="mean")
        print(f"prior MSE: {prior_recon_loss}")

    @torch.no_grad()
    def mse_with_optimal_scale(x, x_hat):
        a = (x * x_hat).mean() / (x_hat * x_hat).mean().clamp(min=1e-8)
        return ((x - a * x_hat) ** 2).mean().item(), a.item()
    print(f"mse_scaled: {mse_with_optimal_scale(images, recon)}")


### Training schedule helper(s)

def set_decoder_trainable(vae, step) -> float:
    if step < 4:
        vae._decoder.train()
        vae._decoder.requires_grad_(True)
        return 0.01
    elif step < 40:
        # Force encoder to minimize loss with a dumb decoder
        # vae._decoder.eval()
        # vae._decoder.requires_grad_(False)
        return 0.001
    else:
        vae._decoder.train()
        vae._decoder.requires_grad_(True)
        return 0.003

### Training loop

def kl_divergence(mu, lv):
    return -0.5 * torch.sum(1 + lv - mu.pow(2) - lv.exp(), dim=1)

if __name__=="__main__":
    run_id = sys.argv[1] if len(sys.argv) > 1 else "default"

    # hyperparams
    epochs = 20_000  # by the time my kids have kids
    global_step = 0
    kl_warmup_steps = 5000
    freeze_step = 4
    unfreeze_step = 40
    beta_step = 400
    batch_size = 128

    # load dynamic training set
    dset = "train2014"
    dataset = CocoImageDataset(DATA_DIR + f"{dset}/{dset}/", count=None)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # load static validation set
    v_dset = "val2014"
    v_dataset = CocoImageDataset(DATA_DIR + f"{v_dset}/{v_dset}/", count=32, rrc=False)
    v_loader = DataLoader(v_dataset, batch_size=32, shuffle=False)

    # torch object creation
    config = VAEConfig() 
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vae = VAE(config).to(device)
    opt = torch.optim.Adam(vae.parameters(), lr=0.001, weight_decay=0.0001)

    log_params(vae)
    print(f"batch size: {batch_size}\n")

    for epoch in range(epochs):
        for images in loader:
            # prepare for step
            vae.train()  
            kl_weight = set_decoder_trainable(vae, global_step)

            # forward
            step_start = perf_counter()
            images = images.to(device)
            recon, mu, lv = vae(images)
            
            # simple normalized recon + weighted KL
            recon_loss = F.mse_loss(recon, images, reduction="mean")  # avg losses
            kl_loss = kl_divergence(mu, lv)
            loss = (recon_loss + kl_weight * kl_loss).mean()

            # backward
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                filter(lambda p: p.requires_grad, vae.parameters()),
                1.0,
            )
            opt.step()

            global_step += 1

            # logging every 10
            if global_step % 10 == 1:
                print(f"\nStep: {global_step}, Loss: {loss} (RL: {recon_loss.mean()}, KL: {kl_loss.mean()}, KLw: {kl_weight})")
                print(f"perf: {perf_counter()-step_start}")
                print(f"mu.mean: {mu.abs().mean().item()}, lv.mean: {lv.mean().item()}")
                print(f"mu.pdist: {torch.pdist(mu).mean().item()}")
                image_stats("x", images)
                image_stats("x_hat", recon)

            # validation set + visualization
            if global_step % 50 == 1:
                vae.eval()
                with torch.no_grad():
                    for v_images in v_loader:
                        v_images = v_images.to(device)
                        v_recon, v_mu, v_lv = vae(v_images)
                        v_recon_loss = F.mse_loss(v_recon, v_images, reduction="mean")
                        v_kl_loss = kl_divergence(v_mu, v_lv)
                        v_recon_mu = vae._decoder(v_mu)
                        print(f"\nValidation: {global_step}, RL: {v_recon_loss.mean()}, KL: {v_kl_loss.mean()})")
                        print(f"mu.mean: {v_mu.abs().mean().item()}, lv.mean: {v_lv.mean().item()}")
                        print(f"mu.pdist: {torch.pdist(v_mu).mean().item()}")

                    def denorm_imagenet(t):
                        mean = torch.tensor(IMAGENET_MEAN, device=t.device).view(1,3,1,1)
                        std  = torch.tensor(IMAGENET_STD,  device=t.device).view(1,3,1,1)
                        return (t * std + mean)
                    image_display = denorm_imagenet(v_images).clamp(0, 1)
                    recon_display = denorm_imagenet(v_recon).clamp(0, 1)
                    recon_mu_display = denorm_imagenet(v_recon_mu).clamp(0, 1)
                    save_x_and_recon(
                        image_display, 
                        recon_display, 
                        recon_mu_display, 
                        filename=run_id+f"/step_{global_step}.png",
                    )
