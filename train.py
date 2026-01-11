"""
training code for vae

new run
> python3 train.py my_fun_run_id

continue from weights (in this case, from step 4000)
> python3 train.py my_fun_run_id 4000

results saved in 
- /logs/<run_id>/logfile.txt
- /logs/<run_id>/step_N.jpg
- /logs/<run_id>/step_N.tar
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
from torchvision.utils import save_image

from model import VariationalAutoEncoder as VAE, VAEConfig


DATA_DIR = "/Users/williamdreese/percy/vqa/VQA/Images/mscoco/"
# torch.autograd.set_detect_anomaly(True) # use for NaN hunting


### Training, eval, & test loading

COLOR_MEAN = (0.485, 0.456, 0.406)
COLOR_STD  = (0.229, 0.224, 0.225)

class CocoImageDataset(Dataset):
    def __init__(self, image_dir, count=None, rrc=True, flip=True):
        self.image_dir = image_dir
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

        if flip:
            trans.extend([
                transforms.RandomHorizontalFlip(p=0.5),
            ])

        trans.extend([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=list(COLOR_MEAN),
                std=list(COLOR_STD),
            ),
        ])
        self.transform = transforms.Compose(trans)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        return image


### Logging & visualization

LOGDIR  = "logs/"
LOGFILE = "logfile.txt"

class Logger:
    def __init__(self, run_id, checkpoint_id):
        self._run_id = run_id
        self._base   = LOGDIR+run_id+"/"
        self._ckpt   = checkpoint_id

        # fail on duplicate run_id for now (unless cont. training)
        if os.path.isfile(self._base+LOGFILE) and not checkpoint_id:
            print("this run_id already exists (np ckpt) - exitting")
            sys.exit(1)

    def log(self, txt):
        print(txt)
        fn = self._base+LOGFILE
        if self._ckpt:
            fn = self._base+f'logfile_from_{self._ckpt}.txt'
        
        # just create new file for logging if it's a continue training run,
        # to avoid dual-writing the same step to a file
        # (let web app read both and dedupe)
        with open(fn, 'a') as f:
            f.write(txt)

def log_params(model, logger):
    total_params, total_bytes = 0, 0
    for name, param in model.named_parameters():
        total_params += param.numel()
        total_bytes += param.nelement() * param.element_size()
    param_size_mb = total_bytes / (1024**2)
    logger.log(f"Count: {total_params:,}")
    logger.log(f"Size (Bytes): {total_bytes}")
    logger.log(f"Size (MB): {param_size_mb:.4f}")

def nan_check(tensor):
    return torch.isnan(tensor).any().item()

def weight_nan_check(model):
    for name, param in model.named_parameters():
        if nan_check(param.data):
            return True
    return False

@torch.no_grad()
def save_triplet(x, x_hat, x_hat_mu, filename, idx=3):
    trip = torch.stack([x[idx], x_hat[idx], x_hat_mu[idx]], dim=0)
    save_image(trip, filename, nrow=3)

def check_explosive_gradients(model):
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            max_grad = param.grad.detach().abs().max()
            if max_grad >= 1e4:
                print(f"Layer: {name} | Max Gradient: {max_grad.item():.4e} <=====")

def image_stats(name, t):
    t = t.detach()
    print(
        f"{name}: shape={tuple(t.shape)} dtype={t.dtype} "
        f"min={t.min().item():.4f} max={t.max().item():.4f} mean={t.mean().item():.4f}"
    )


### Training schedule helper(s)

def set_decoder_trainable(vae, step) -> float:
    # linear, but need to get this schedule down
    # or maybe account for R-D curve
    return min(max(step / 100_000, 0.001), 0.02)

### Training loop

def kl_divergence(mu, lv):
    return -0.5 * torch.sum(1 + lv - mu.pow(2) - lv.exp(), dim=1)

if __name__=="__main__":
    run_id = sys.argv[1] if len(sys.argv) > 1 else "default"
    checkpoint_id = int(sys.argv[2]) if len(sys.argv) > 2 else None

    # hyperparams
    epochs = 20_000  # by the time my kids have kids
    global_step = 0
    batch_size = 128

    # load dynamic training set
    dset = "train2014"
    dataset = CocoImageDataset(DATA_DIR + f"{dset}/{dset}/")
    loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=5,
        persistent_workers=True,
        prefetch_factor=4,
    )

    # uncomment for overfit testing
    # dataset = CocoImageDataset(DATA_DIR + f"{dset}/{dset}/", count=8, rrc=False, flip=False)
    # loader = DataLoader(dataset, batch_size=8, shuffle=False)

    # load static validation set
    v_dset = "val2014"
    v_dataset = CocoImageDataset(DATA_DIR + f"{v_dset}/{v_dset}/", count=32, rrc=False)
    v_loader = DataLoader(
        v_dataset, 
        batch_size=32, 
        shuffle=False,
        num_workers=1,
        persistent_workers=True,
    )

    # cpu performance guidance
    torch.set_num_threads(os.cpu_count()-6)
    torch.set_num_interop_threads(1)

    # torch object creation
    config = VAEConfig() 
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vae = VAE(config).to(device)
    opt = torch.optim.Adam(vae.parameters(), lr=0.001, weight_decay=0.0001)

    if checkpoint_id:
        ckpt_file = f'logs/{run_id}/step_{checkpoint_id}.tar'
        checkpoint = torch.load(ckpt_file, weights_only=True)
        vae.load_state_dict(checkpoint['model_state_dict'])
        opt.load_state_dict(checkpoint['optimizer_state_dict'])
        global_step = checkpoint['global_step']

    logger = Logger(run_id, checkpoint_id)
    log_params(vae, logger)
    logger.log(f"batch size: {batch_size}\n")
    step_start = perf_counter()

    for epoch in range(epochs):
        vae.train()
        for images in loader:
            # prepare for step
            kl_weight = set_decoder_trainable(vae, global_step)

            # forward
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
                step_end = perf_counter()
                logger.log(f"\nStep: {global_step}, Loss: {loss.detach()} (RL: {recon_loss.mean().detach()}, KL: {kl_loss.mean().detach()}, KLw: {kl_weight})")
                logger.log(f"10-step perf: {step_end-step_start}")
                logger.log(f"mu.mean: {mu.abs().mean().detach()}, lv.mean: {lv.mean().detach()}")
                step_start = step_end  # reset loop timer (includes val & weight save)
                # too expensive
                # logger.log(f"mu.pdist: {torch.pdist(mu).mean().item()}")

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
                        logger.log(f"\nValidation: {global_step}, RL: {v_recon_loss.mean()}, KL: {v_kl_loss.mean()})")
                        logger.log(f"mu.mean: {v_mu.abs().mean().item()}, lv.mean: {v_lv.mean().item()}")
                        # logger.log(f"mu.pdist: {torch.pdist(v_mu).mean().item()}")

                    def denorm_imagenet(t):
                        mean = torch.tensor(COLOR_MEAN, device=t.device).view(1,3,1,1)
                        std  = torch.tensor(COLOR_STD,  device=t.device).view(1,3,1,1)
                        return (t * std + mean)
                    image_display = denorm_imagenet(v_images).clamp(0, 1)
                    recon_display = denorm_imagenet(v_recon).clamp(0, 1)
                    recon_mu_display = denorm_imagenet(v_recon_mu).clamp(0, 1)
                    # save_x_and_recon(
                    save_triplet(    
                        image_display, 
                        recon_display, 
                        recon_mu_display, 
                        filename="logs/"+run_id+f"/step_{global_step}.png",
                    )
                vae.train()
            # save weights for future testing/ training/ probing
            if global_step % 1000 == 1 and global_step != 1:
                checkpoint_path = f'logs/{run_id}/step_{global_step}.tar'
                torch.save({
                    'global_step': global_step,
                    'model_state_dict': vae.state_dict(),
                    'optimizer_state_dict': opt.state_dict(),
                }, checkpoint_path)
