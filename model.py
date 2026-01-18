"""
define models used in project
"""
import torch
import torch.nn as nn

class VAEConfig():
    def __init__(self):
        self.x_w = 224
        self.x_h = 224
        self.feature_w = 7
        self.feature_h = 7
        self.latent_dim = 256


def nan_check(tensor):
    return torch.isnan(tensor).any().item()


"""
Baseline convolutional blocks 
for both encoding & decoding latent spaces
"""
class ConvBlock(nn.Module):
    def __init__(self, ic, oc, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self._block = nn.Sequential(
            nn.Conv2d(ic, oc, kernel_size, stride, padding),
            nn.BatchNorm2d(oc),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self._block(x)

class ConvUpBlock(nn.Module):
    def __init__(self, ic, oc, kernel_size=4, stride=2, padding=1, act=nn.ReLU(inplace=True)):
        super().__init__()
        self._block = nn.Sequential(
            # try mode="bilinear" as well some time
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(ic, oc, kernel_size=3, padding=1),
            nn.BatchNorm2d(oc),
            act,
        )

    def forward(self, x):
        return self._block(x)

class ConvUpBlock_NoBN(nn.Module):
    def __init__(self, ic, oc, kernel_size=4, stride=2, padding=1, act=nn.ReLU(inplace=True)):
        super().__init__()
        self._block = nn.Sequential(
            # try mode="bilinear" as well some time
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(ic, oc, kernel_size=3, padding=1),
            act,
        )

    def forward(self, x):
        return self._block(x)

"""
Baseline Encoder/decoder abstractions
"""
class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self._config = config

        # half-vgg 19
        self._encoder = nn.Sequential(
            ConvBlock(3, 64),  # 224x224
            nn.MaxPool2d(2),   # 112x112
            ConvBlock(64, 128),
            nn.MaxPool2d(2),   # 56x56
            ConvBlock(128, 256),
            nn.MaxPool2d(2),   # 28x28
            ConvBlock(256, 256),
            nn.MaxPool2d(2),   # 14x14
            ConvBlock(256, 256),
            nn.MaxPool2d(2),   # 7x7
        )

    def forward(self, x):
        return self._encoder(x)


class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self._config = config

        # latent comes with spatial structure - no need to recreate
        # seed does not upsample, but preps signal for upsampling
        self._seed = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        
        # focused on upsampling into a real RGB image
        self._decoder = nn.Sequential(
            ConvUpBlock_NoBN(128, 96, act=nn.Identity()),  # 7
            nn.Conv2d(96, 96, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            ConvUpBlock_NoBN(96, 64),   # 14
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            ConvUpBlock_NoBN(64, 32),   # 28 
            ConvUpBlock_NoBN(32, 16),   # 56
            ConvUpBlock_NoBN(16, 3, act=nn.Identity()),    # 112
            # nn.Conv2d(16, 3, kernel_size=3, padding=1), # still 224x224
        )

    def forward(self, z):
        h = self._seed(z) 
        return self._decoder(h)


class SpatialPosteriorHead(nn.Module):
    def __init__(self, in_channels: int, latent_dim: int):
        super().__init__()
        self.mu_conv = nn.Conv2d(in_channels, latent_dim, kernel_size=1)
        self.logvar_conv = nn.Conv2d(in_channels, latent_dim, kernel_size=1)

    def forward(self, h):  # [B, C, H, W]
        mu = self.mu_conv(h)         # [B, D, H, W]
        logvar = self.logvar_conv(h) # [B, D, H, W]
        return mu, logvar

"""
Full x -> x' model 
"""
class VariationalAutoEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self._config = config

        # [B,3,W,H] -> [B,C,7,7]
        self._encoder = Encoder(config)

        # Learned (spatial) posterior weights
        # [B,C,7,7] -> [B,D,7,7]
        self._post_head = SpatialPosteriorHead(256, 16)

        # [B,C,7,7] -> [B,3,W,H]
        self._decoder = Decoder(config)

    def forward(self, x):
        # grab some 2d features
        features = self._encoder(x)

        # don't flatten - use spatial latent head
        mu, lv = self._post_head(features)
        clipped_lv = torch.clamp(lv, min=-10.0, max=1.0)  # help combat exploding gradients
        std = torch.exp(clipped_lv / 2.0)
        sample = mu  + (torch.randn_like(std) * std)  # reparam trick

        x_hat = self._decoder(sample)
        return x_hat, sample, mu, clipped_lv


if __name__=="__main__":
    vae = VariationalAutoEncoder(VAEConfig())
    total_params, total_bytes = 0, 0
    for name, param in vae.named_parameters():
        # Count of elements (parameters)
        param_count = param.numel()
        total_params += param_count
        
        # Size in bytes
        param_size_bytes = param.nelement() * param.element_size()
        total_bytes += param_size_bytes

    print(f"Count: {total_params:,}")
    print(f"Size (Bytes): {total_bytes}")
    param_size_mb = total_bytes / (1024**2)
    print(f"Size (MB): {param_size_mb:.4f}\n")
