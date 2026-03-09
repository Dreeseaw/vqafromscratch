"""
training code for vae

new run
> python3 -m train.train my_fun_run_id

continue from weights (in this case, from step 4000)
> python3 -m train.train my_fun_run_id 4000

results saved in 
- /logs/<run_id>/logfile.txt
- /logs/<run_id>/step_N.jpg
- /logs/<run_id>/step_N.tar
"""
import argparse
from contextlib import nullcontext
import os
import sys
import datetime
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

from models.vae import VariationalAutoEncoder as VAE, VAEConfig
from models.vae import VariationalAutoEncoderRes as VAEr
from models.vae import ViTVAE, ViTVAE2


DEFAULT_DATA_DIR = "images"
SEED = 35
DEFAULT_PRESET = "vitvae2_current"
TRAINING_PRESETS = {
    # Current hardcoded behavior before CLI presets were introduced.
    "vitvae2_current": {
        "model": "vitvae2",
        "latent_dim": 768,
        "cbld": 1536,
        "epochs": 20_000,
        "batch_size": 128,
        "num_workers": 4,
        "prefetch_factor": 4,
        "lr": 0.001,
        "weight_decay": 0.0001,
        "alpha": 0.0,
        "beta": 0.00521,
        "gamma": 0.0,
        "corr_reg": True,
        "precision": "fp32",
        "channels_last": False,
        "compile": False,
        "pin_memory": False,
        "log_every": 10,
        "val_every": 50,
        "ckpt_every": 2000,
        "grad_clip": 1.0,
    },
    # Best pre-LM MPS-era gamma-only decorrelation setup recovered from git history.
    "gamma_corr_only": {
        "model": "vaer",
        "latent_dim": 256,
        "cbld": None,
        "epochs": 20_000,
        "batch_size": 96,
        "num_workers": 1,
        "prefetch_factor": 2,
        "lr": 0.001,
        "weight_decay": 0.0001,
        "alpha": 0.0,
        "beta": 0.0005,
        "gamma": 0.001,
        "corr_reg": True,
        "precision": "fp32",
        "channels_last": False,
        "compile": False,
        "pin_memory": False,
        "log_every": 10,
        "val_every": 50,
        "ckpt_every": 2000,
        "grad_clip": 1.0,
    },
    # Throughput-oriented CUDA variant of the same training regime.
    "gamma_corr_only_cuda_fast": {
        "model": "vaer",
        "latent_dim": 256,
        "cbld": None,
        "epochs": 20_000,
        "batch_size": 256,
        "num_workers": 8,
        "prefetch_factor": 8,
        "lr": 0.0005,
        "weight_decay": 0.0001,
        "alpha": 0.0,
        "beta": 0.0005,
        "gamma": 0.001,
        "corr_reg": True,
        "precision": "bf16",
        "channels_last": True,
        "compile": False,
        "pin_memory": True,
        "log_every": 10,
        "val_every": 200,
        "ckpt_every": 4000,
        "grad_clip": 1.0,
    },
}


def _value_or(value, fallback):
    return fallback if value is None else value


def parse_args():
    parser = argparse.ArgumentParser(description="Train a VAE on MSCOCO images.")
    parser.add_argument("run_id", type=str)
    parser.add_argument("checkpoint", nargs="?", type=int)
    parser.add_argument(
        "--preset",
        type=str,
        default=DEFAULT_PRESET,
        choices=sorted(TRAINING_PRESETS.keys()),
        help="Named training preset. CLI flags override the preset.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
    )
    parser.add_argument("--data_dir", type=str, default=DEFAULT_DATA_DIR)
    parser.add_argument("--train_split", type=str, default="train2014")
    parser.add_argument("--val_split", type=str, default="val2014")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--num_workers", type=int)
    parser.add_argument("--prefetch_factor", type=int)
    parser.add_argument("--val_count", type=int, default=32)
    parser.add_argument(
        "--model",
        type=str,
        choices=["vae", "vaer", "vitvae", "vitvae2"],
    )
    parser.add_argument("--latent_dim", type=int)
    parser.add_argument("--cbld", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--weight_decay", type=float)
    parser.add_argument("--alpha", type=float)
    parser.add_argument("--beta", type=float)
    parser.add_argument("--gamma", type=float)
    parser.add_argument("--precision", type=str, choices=["fp32", "bf16", "fp16"])
    parser.add_argument("--log_every", type=int)
    parser.add_argument("--val_every", type=int)
    parser.add_argument("--ckpt_every", type=int)
    parser.add_argument("--grad_clip", type=float)
    parser.add_argument("--corr_reg", dest="corr_reg", action="store_true")
    parser.add_argument("--no_corr_reg", dest="corr_reg", action="store_false")
    parser.add_argument("--channels_last", dest="channels_last", action="store_true")
    parser.add_argument("--no_channels_last", dest="channels_last", action="store_false")
    parser.add_argument("--compile", dest="compile", action="store_true")
    parser.add_argument("--no_compile", dest="compile", action="store_false")
    parser.add_argument("--pin_memory", dest="pin_memory", action="store_true")
    parser.add_argument("--no_pin_memory", dest="pin_memory", action="store_false")
    parser.set_defaults(corr_reg=None)
    parser.set_defaults(channels_last=None, compile=None, pin_memory=None)
    return parser.parse_args()


def resolve_training_config(args):
    preset = TRAINING_PRESETS[args.preset]
    return {
        "model": _value_or(args.model, preset["model"]),
        "latent_dim": _value_or(args.latent_dim, preset["latent_dim"]),
        "cbld": _value_or(args.cbld, preset["cbld"]),
        "epochs": _value_or(args.epochs, preset["epochs"]),
        "batch_size": _value_or(args.batch_size, preset["batch_size"]),
        "num_workers": _value_or(args.num_workers, preset["num_workers"]),
        "prefetch_factor": _value_or(args.prefetch_factor, preset["prefetch_factor"]),
        "lr": _value_or(args.lr, preset["lr"]),
        "weight_decay": _value_or(args.weight_decay, preset["weight_decay"]),
        "alpha": _value_or(args.alpha, preset["alpha"]),
        "beta": _value_or(args.beta, preset["beta"]),
        "gamma": _value_or(args.gamma, preset["gamma"]),
        "corr_reg": _value_or(args.corr_reg, preset["corr_reg"]),
        "precision": _value_or(args.precision, preset["precision"]),
        "channels_last": _value_or(args.channels_last, preset["channels_last"]),
        "compile": _value_or(args.compile, preset["compile"]),
        "pin_memory": _value_or(args.pin_memory, preset["pin_memory"]),
        "log_every": _value_or(args.log_every, preset["log_every"]),
        "val_every": _value_or(args.val_every, preset["val_every"]),
        "ckpt_every": _value_or(args.ckpt_every, preset["ckpt_every"]),
        "grad_clip": _value_or(args.grad_clip, preset["grad_clip"]),
    }


def resolve_device(requested):
    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    if requested == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but not available.")
    if requested == "mps" and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        raise SystemExit("MPS requested but not available.")
    return requested


def build_model(model_name, latent_dim, cbld, device):
    config = VAEConfig(latent_dim=latent_dim, cbld=cbld)
    if model_name == "vae":
        model = VAE(config)
    elif model_name == "vaer":
        model = VAEr(config)
    elif model_name == "vitvae":
        model = ViTVAE(config)
    elif model_name == "vitvae2":
        model = ViTVAE2(config)
    else:
        raise ValueError(f"unknown model: {model_name}")
    return model.to(device)


def resolve_split_dir(data_dir, split):
    candidates = [
        os.path.join(data_dir, split),
        os.path.join(data_dir, split, split),
    ]
    for path in candidates:
        if os.path.isdir(path):
            return path
    raise FileNotFoundError(
        f"could not find image split '{split}' under '{data_dir}'. "
        f"Tried: {', '.join(candidates)}"
    )


def resolve_amp_dtype(device, precision):
    if device != "cuda" or precision == "fp32":
        return None
    if precision == "bf16":
        return torch.bfloat16
    if precision == "fp16":
        return torch.float16
    raise ValueError(f"unknown precision: {precision}")


def autocast_ctx(device, amp_dtype):
    if device == "cuda" and amp_dtype is not None:
        return torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=True)
    return nullcontext()


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
    def __init__(self, run_id, checkpoint_id, probe=False):
        self._run_id = run_id
        self._base   = LOGDIR+run_id+"/"
        self._ckpt   = checkpoint_id
        self._probe  = probe

        # fail on duplicate run_id for now (unless cont. training)
        if os.path.isfile(self._base+LOGFILE) and not checkpoint_id and not probe:
            print("this run_id already exists (np ckpt) - exitting")
            sys.exit(1)

        self._fn = self._base+LOGFILE
        if self._probe:
            self._fn = self._base+f'logfile_probe{self._ckpt}.txt'
        elif self._ckpt:
            self._fn = self._base+f'logfile_from_{self._ckpt}.txt'

    def log(self, txt):
        print(txt)
        
        # just create new file for logging if it's a continue training run,
        # to avoid dual-writing the same step to a file
        # (let web app read both and dedupe)
        with open(self._fn, 'a') as f:
            f.write(txt)

def log_params(model, logger):
    total_params, total_bytes = 0, 0
    for name, param in model.named_parameters():
        total_params += param.numel()
        total_bytes += param.nelement() * param.element_size()
    enc_params, enc_bytes = 0, 0
    for name, param in model._encoder.named_parameters():
        enc_params += param.numel()
        enc_bytes += param.nelement() * param.element_size()
    dec_params, dec_bytes = 0, 0
    for name, param in model._decoder.named_parameters():
        dec_params += param.numel()
        dec_bytes += param.nelement() * param.element_size()
    param_size_mb = total_bytes / (1024**2)
    enc_param_size_mb = enc_bytes / (1024**2)
    dec_param_size_mb = dec_bytes / (1024**2)
    latent_param_size_mb = param_size_mb - (enc_param_size_mb + dec_param_size_mb)
    logger.log(f"Total: {total_params:,}")
    logger.log(f"Total size (MB): {param_size_mb:.4f}")
    logger.log(f"Encoder size (MB): {enc_param_size_mb:.4f}")
    logger.log(f"Decoder size (MB): {dec_param_size_mb:.4f}")
    logger.log(f"Latent head size (MB): {latent_param_size_mb:.4f}")

def nan_check(tensor):
    return torch.isnan(tensor).any().item()

def weight_nan_check(model):
    for name, param in model.named_parameters():
        if nan_check(param.data):
            return True
    return False

@torch.no_grad()
def save_quadlet(x, x_hat, x_hat_mu, kl_hm, filename, idx=3):
    eps = 1e-8
    hm = kl_hm[idx:idx+1].unsqueeze(1)
    hm_up = F.interpolate(hm, size=(224,224), mode="nearest")
    
    # sclaes for coloring heatmap
    h = hm_up[0, 0]  # [H,W]
    h_min = h.min()
    h_max = h.max()
    h01 = (h - h_min) / (h_max - h_min + eps)  # [H,W], 0=unused, 1=most used

    # 3) blue->red colormap: (R,G,B) = (t, 0, 1-t)
    red = h01
    green = torch.zeros_like(h01)
    blue = 1.0 - h01
    hm_rgb = torch.stack([red, green, blue], dim=0)

    quad = torch.stack([x[idx], x_hat[idx], x_hat_mu[idx], hm_rgb], dim=0)
    save_image(quad, filename, nrow=4)

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

def spatial_latent_stats(mu, lv, val=False):
    """
    calculate some latent space variance/ activity metrics

    inp: 
        mu: [B,C,H,W]
        lv: [B,C,H,W]
    out: 
        # of active units: N
        # of active KL units: N
        activation heatmap: [B,H,W]
    """

    # mu-active unit counting
    tau = 0.001
    au_map = mu.var(dim=0, unbiased=False)
    active_frac = (au_map > tau).float().mean().item()

    # kl-active unit counting
    kl_elems = 0.5 * (mu**2 + lv.exp() - 1.0 - lv)
    kl_active_map = (kl_elems > tau).float().flatten(1).mean(1)
    kl_active_frac = kl_active_map.mean().item()

    # only visualize on val
    if not val:
        return (
            active_frac,
            kl_active_frac,
            None,
            None
        )

    # heatmap calc w/ relative usage
    eps = 1e-8
    kl_hw = kl_elems.sum(dim=1)
    kl_hw_norm = kl_hw / (kl_hw.flatten(1).mean(dim=1)[:,None,None] + eps)

    # signal-to-noise heatmap
    std = torch.exp(0.5 * lv)
    snr_hw = (mu.abs() / (std + eps)).mean(dim=1)
    snr_hw_norm = snr_hw / (snr_hw.flatten(1).mean(dim=1)[:,None,None] + eps)

    return (
        active_frac, 
        kl_active_frac, 
        kl_hw_norm,
        snr_hw_norm
    )


### Loss Helpers

def kl_divergence(mu, lv):
    return -0.5 * torch.sum(1 + lv - mu.pow(2) - lv.exp(), dim=1)

def _pairwise_sq_dists(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # x: [B,D], y: [B,D] -> [B,B]
    x2 = (x * x).sum(dim=1, keepdim=True)          # [B,1]
    y2 = (y * y).sum(dim=1, keepdim=True).T        # [1,B]
    return x2 + y2 - 2.0 * (x @ y.T)

def mmd_rbf(x: torch.Tensor, y: torch.Tensor, sigmas=(1, 2, 4, 8, 16)) -> torch.Tensor:
    """
    Unbiased MMD^2 with a mixture of RBF kernels.
    x, y: [B,D]
    """
    B = x.size(0)
    xx = _pairwise_sq_dists(x, x)
    yy = _pairwise_sq_dists(y, y)
    xy = _pairwise_sq_dists(x, y)

    Kxx = 0.0
    Kyy = 0.0
    Kxy = 0.0
    for s in sigmas:
        gamma = 1.0 / (2.0 * (s ** 2))
        Kxx = Kxx + torch.exp(-gamma * xx)
        Kyy = Kyy + torch.exp(-gamma * yy)
        Kxy = Kxy + torch.exp(-gamma * xy)

    # remove diagonal for unbiased estimate
    Kxx = Kxx - torch.diag(torch.diag(Kxx))
    Kyy = Kyy - torch.diag(torch.diag(Kyy))

    mmd = Kxx.sum() / (B * (B - 1)) + Kyy.sum() / (B * (B - 1)) - 2.0 * Kxy.mean()
    return mmd

def mmd_imq(x: torch.Tensor, y: torch.Tensor, scales=(0.1, 0.2, 0.5, 1.0, 2.0)) -> torch.Tensor:
    """
    Unbiased MMD^2 with IMQ kernel: k(a,b)=C/(C+||a-b||^2).
    Often works well in practice for VAEs.
    """
    B = x.size(0)
    xx = _pairwise_sq_dists(x, x)
    yy = _pairwise_sq_dists(y, y)
    xy = _pairwise_sq_dists(x, y)

    Kxx = 0.0
    Kyy = 0.0
    Kxy = 0.0
    for C in scales:
        Kxx = Kxx + (C / (C + xx))
        Kyy = Kyy + (C / (C + yy))
        Kxy = Kxy + (C / (C + xy))

    Kxx = Kxx - torch.diag(torch.diag(Kxx))
    Kyy = Kyy - torch.diag(torch.diag(Kyy))

    mmd = Kxx.sum() / (B * (B - 1)) + Kyy.sum() / (B * (B - 1)) - 2.0 * Kxy.mean()
    return mmd


def orthogonal_reg(mu: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    mu: (B, ...) encoder means
    returns: scalar decorrelation loss
    """
    mu = mu.flatten(1)
    (B, D) = mu.shape

    # center per latent dim
    mu_centered = mu - mu.mean(dim=0, keepdim=True)

    # covariance (D x D)
    cov = (mu_centered.T @ mu_centered) / (B + eps)

    # off-diagonal penalty
    off_diag = cov - torch.diag(torch.diag(cov))

    loss = (off_diag ** 2).sum()
    return loss


def orthogonal_reg_spatial(mu: torch.Tensor, corr: bool = True, eps: float = 1e-8) -> torch.Tensor:
    """
    Computes channel covariance (or corr) over all batch+spatial samples (N = B*H*W),
    then penalizes off-diagonal entries.

    mu: [B,C,H,W] tensor
    corr: if true, variance is normalized (helps with lv collapse)

    returns: scalar decorrelation loss
    """

    B, C, H, W = mu.shape
    x = mu.permute(0, 2, 3, 1).reshape(-1, C)
    x = x - x.mean(dim=0, keepdim=True)

    # normalize per channel for correlation
    if corr:
        std = x.std(dim=0, keepdim=True).clamp_min(eps)
        x = x / std

    N = x.shape[0]
    corr = (x.T @ x) / (N + eps)

    # exclude diagonal (i==j)
    off = corr - torch.diag(torch.diag(corr))
    return (off ** 2).sum()


### Training schedule helper(s)

def get_loss_weights(step, alpha, beta, gamma) -> (float, float, float):
    """
    Fixed loss weights for now; the helper remains here so schedules can be restored later.
    returns:
        alpha (float): term for mmd weight
        beta (float): term for kl weight (Original VAE paper B = 0.00521 due to normalization)
        gamma (float): term for ortho reg weight
    """
    return (alpha, beta, gamma)


### Training loop

if __name__=="__main__":
    args = parse_args()
    train_cfg = resolve_training_config(args)
    run_id = args.run_id
    checkpoint_id = args.checkpoint

    # hyperparams
    epochs = train_cfg["epochs"]  # by the time my kids have kids
    global_step = 0
    batch_size = train_cfg["batch_size"]

    # cpu performance guidance
    device = resolve_device(args.device)
    if device == "cuda":
        torch.backends.cudnn.benchmark = True
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")

    if device == "cpu":
        torch.set_num_threads(8)
        torch.set_num_interop_threads(1)

    # Set RNG seed
    torch.manual_seed(SEED)

    # load dynamic training set
    dset = args.train_split
    train_dir = resolve_split_dir(args.data_dir, dset)
    dataset = CocoImageDataset(train_dir)
    train_pin_memory = bool(train_cfg["pin_memory"]) and device == "cuda"
    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": True,
        "num_workers": train_cfg["num_workers"],
        "pin_memory": train_pin_memory,
    }
    if train_cfg["num_workers"] > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = train_cfg["prefetch_factor"]
    loader = DataLoader(dataset, **loader_kwargs)

    # uncomment for overfit testing
    # dataset = CocoImageDataset(train_dir, count=8, rrc=False, flip=False)
    # loader = DataLoader(dataset, batch_size=8, shuffle=False)

    # load static validation set
    v_dset = args.val_split
    val_dir = resolve_split_dir(args.data_dir, v_dset)
    val_pin_memory = train_pin_memory
    v_dataset = CocoImageDataset(
        val_dir,
        count=args.val_count,
        rrc=False,
    )
    v_loader = DataLoader(
        v_dataset, 
        batch_size=args.val_count,
        shuffle=False,
        pin_memory=val_pin_memory,
    )

    # torch object creation
    vae = build_model(
        train_cfg["model"],
        train_cfg["latent_dim"],
        train_cfg["cbld"],
        device,
    )
    if device == "cuda" and train_cfg["channels_last"]:
        vae = vae.to(memory_format=torch.channels_last)
    opt = torch.optim.Adam(
        vae.parameters(),
        lr=train_cfg["lr"],
        weight_decay=train_cfg["weight_decay"],
    )
    amp_dtype = resolve_amp_dtype(device, train_cfg["precision"])
    use_grad_scaler = device == "cuda" and train_cfg["precision"] == "fp16"
    grad_scaler = torch.cuda.amp.GradScaler(enabled=use_grad_scaler)

    if checkpoint_id:
        ckpt_file = f'logs/{run_id}/step_{checkpoint_id}.tar'
        checkpoint = torch.load(ckpt_file, weights_only=True)
        vae.load_state_dict(checkpoint['model_state_dict'])
        opt.load_state_dict(checkpoint['optimizer_state_dict'])
        global_step = checkpoint['global_step']
    if train_cfg["compile"]:
        if not hasattr(torch, "compile"):
            raise SystemExit("torch.compile requested but not available in this PyTorch build.")
        vae = torch.compile(vae)

    logger = Logger(run_id, checkpoint_id)
    log_params(vae, logger)
    logger.log(f"preset: {args.preset}")
    logger.log(f"data_dir: {args.data_dir}")
    logger.log(f"train_dir: {train_dir}")
    logger.log(f"val_dir: {val_dir}")
    logger.log(
        "runtime: "
        f"precision={train_cfg['precision']}, "
        f"pin_memory={train_pin_memory}, "
        f"channels_last={train_cfg['channels_last']}, "
        f"compile={train_cfg['compile']}"
    )
    logger.log(
        "model: "
        f"{train_cfg['model']} "
        f"(latent_dim={train_cfg['latent_dim']}, cbld={train_cfg['cbld']})"
    )
    logger.log(
        "loss weights: "
        f"alpha={train_cfg['alpha']}, beta={train_cfg['beta']}, "
        f"gamma={train_cfg['gamma']}, corr_reg={train_cfg['corr_reg']}"
    )
    logger.log(
        "cadence: "
        f"log_every={train_cfg['log_every']}, "
        f"val_every={train_cfg['val_every']}, "
        f"ckpt_every={train_cfg['ckpt_every']}"
    )
    logger.log(f"batch size: {batch_size}")
    logger.log(f"Run start time: {str(datetime.datetime.now())}")
    logger.log(f"Running on {device}\n")
    step_start = perf_counter()

    for epoch in range(epochs):
        vae.train()
        for images in loader:
            # prepare for step
            (alpha, beta, gamma) = get_loss_weights(
                global_step,
                train_cfg["alpha"],
                train_cfg["beta"],
                train_cfg["gamma"],
            )

            # forward
            images = images.to(device, non_blocking=train_pin_memory)
            if device == "cuda" and train_cfg["channels_last"]:
                images = images.contiguous(memory_format=torch.channels_last)
            with autocast_ctx(device, amp_dtype):
                recon, z, mu, lv = vae(images)

            # keep the model fast under autocast, but do fragile VAE loss math in fp32
            recon_f = recon.float()
            z_f = z.float()
            mu_f = mu.float()
            lv_f = lv.float()
            images_f = images.float()

            recon_loss = F.mse_loss(recon_f, images_f, reduction="mean")
            kl_loss = kl_divergence(mu_f, lv_f)
            mmd = mmd_imq(z_f.flatten(1), torch.randn_like(z_f).flatten(1))
            ortho = orthogonal_reg_spatial(mu_f, corr=train_cfg["corr_reg"])
            loss = (recon_loss + beta * kl_loss).mean() + (alpha * mmd) + (gamma * ortho)
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"non-finite loss at step {global_step + 1}: "
                    f"loss={loss.detach().item()} "
                    f"recon={recon_loss.detach().item()} "
                    f"kl={kl_loss.mean().detach().item()} "
                    f"mmd={mmd.detach().item()} "
                    f"ortho={ortho.detach().item()}"
                )

            # backward
            opt.zero_grad(set_to_none=True)
            if use_grad_scaler:
                grad_scaler.scale(loss).backward()
                grad_scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(
                    filter(lambda p: p.requires_grad, vae.parameters()),
                    train_cfg["grad_clip"],
                )
                grad_scaler.step(opt)
                grad_scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    filter(lambda p: p.requires_grad, vae.parameters()),
                    train_cfg["grad_clip"],
                )
                opt.step()

            global_step += 1

            # logging every 10
            if global_step % train_cfg["log_every"] == 1:
                step_end = perf_counter()
                (ad, akld, _, _) = spatial_latent_stats(mu_f, lv_f, val=False)
                logger.log(
                    f"\nStep: {global_step}, Loss: {loss.detach():.4f} "
                    f"(RL: {recon_loss.mean().detach():.4f}, "
                    f"KL: {kl_loss.mean().detach():.4f}, KLw: {beta}, "
                    f"MMD: {mmd.detach():.7f} MMDw: {alpha})"
                )
                logger.log(
                    f"{train_cfg['log_every']}-step im/s: "
                    f"{((batch_size * train_cfg['log_every']) / (step_end-step_start)):.4f}"
                )
                logger.log(f"mu.mean: {mu.flatten(1).abs().mean().detach():.4f}, lv.mean: {lv.flatten(1).mean().detach():.4f}")
                logger.log(f"Active mu dims: {ad:.4f}, Active KL dims: {akld:.4f}")
                logger.log(f"Ortho reg ({gamma}): {ortho.detach():.4f}")
                logger.log(f"mu2.mean: {(mu.flatten(1) ** 2).mean().detach():.4f}, var.mean: {torch.exp(lv).mean().detach():.4f}")
                step_start = step_end  # reset loop timer (includes val & weight save)

            # validation set + visualization every 50
            if train_cfg["val_every"] > 0 and global_step % train_cfg["val_every"] == 1:
                vae.eval()
                with torch.no_grad():
                    for v_images in v_loader:
                        v_images = v_images.to(device, non_blocking=val_pin_memory)
                        if device == "cuda" and train_cfg["channels_last"]:
                            v_images = v_images.contiguous(memory_format=torch.channels_last)
                        with autocast_ctx(device, amp_dtype):
                            v_recon, v_sample, v_mu, v_lv = vae(v_images)
                            v_recon_mu = vae._decoder(v_mu)
                        v_recon_loss = F.mse_loss(v_recon.float(), v_images.float(), reduction="mean")
                        v_kl_loss = kl_divergence(v_mu.float(), v_lv.float())
                        (ad, akld, kl_hm, snr_hm) = spatial_latent_stats(mu_f, lv_f, val=True)
                        logger.log(f"\nValidation: {global_step}, RL: {v_recon_loss.mean().detach():.4f}, KL: {v_kl_loss.mean().detach():.4f})")
                        logger.log(f"mu.mean: {v_mu.flatten(1).abs().mean().detach():.4f}, lv.mean: {v_lv.flatten(1).mean().detach():.4f}")
                        logger.log(f"Active mu dims: {ad:.4f}, Active KL dims: {akld:.4f}")

                    def denorm_imagenet(t):
                        mean = torch.tensor(COLOR_MEAN, device=t.device).view(1,3,1,1)
                        std  = torch.tensor(COLOR_STD,  device=t.device).view(1,3,1,1)
                        return (t * std + mean)
                    image_display = denorm_imagenet(v_images).clamp(0, 1)
                    recon_display = denorm_imagenet(v_recon).clamp(0, 1)
                    recon_mu_display = denorm_imagenet(v_recon_mu).clamp(0, 1)
                    pic_fn = "logs/"+run_id+f"/step_{global_step}.png"
                    # don't overwrite old photos
                    if checkpoint_id:
                        pic_fn = "logs/"+run_id+f"/step_{global_step}_from_{checkpoint_id}.png"
                    save_quadlet(    
                        image_display, 
                        recon_display, 
                        recon_mu_display, 
                        kl_hm,
                        filename=pic_fn,
                    )
                vae.train()

            # save weights for future testing/ training/ probing every 1000
            if (
                train_cfg["ckpt_every"] > 0
                and global_step % train_cfg["ckpt_every"] == 1
                and global_step != 1
            ):
                checkpoint_path = f'logs/{run_id}/step_{global_step}.tar'
                if checkpoint_id:
                    checkpoint_path = f'logs/{run_id}/step_{global_step}_from_{checkpoint_id}.tar'
                torch.save({
                    'global_step': global_step,
                    'model_state_dict': vae.state_dict(),
                    'optimizer_state_dict': opt.state_dict(),
                }, checkpoint_path)
