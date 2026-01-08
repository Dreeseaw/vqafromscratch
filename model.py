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

        # get spatial info back
        
        # focused on extracting spatial info
        self._spatialer = nn.Linear(config.latent_dim, 64 * 14 * 14)
        # tensor gets shape-change in between layers 
        self._precoder = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # nn.Conv2d(128, 256, kernel_size=3, padding=1),
            # nn.ReLU(inplace=True),
        )
        
        # focused on upsampling into a real RGB image
        self._decoder = nn.Sequential(
            ConvUpBlock_NoBN(128, 96),  # 14x14 (x2)
            ConvUpBlock_NoBN(96, 64),   # 28x28 (x2)
            ConvUpBlock_NoBN(64, 32),    # 56x56 (x2)
            ConvUpBlock_NoBN(32, 16),    # 112x112 (x2)
            nn.Conv2d(16, 3, kernel_size=3, padding=1), # still 224x224
        )

    def forward(self, z):
        h = self._spatialer(z)
        h = h.view(
            z.size(0), 
            self._config.latent_dim // 4, 
            self._config.feature_w * 2,
            self._config.feature_h * 2,
        )
        h = self._precoder(h) 
        return self._decoder(h)


"""
Full x -> x' model 
"""
class VariationalAutoEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self._config = config
        self._flat_feature_dim = config.latent_dim * config.feature_w * config.feature_h

        # [B,W,H,3] -> [B,D]
        self._encoder = Encoder(config)
        encoder_output_channels = 256  # keep here in case it changes a lot
        # self._gap = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        # Learned posterior weights
        self._mu = nn.Linear(encoder_output_channels * 7 * 7, self._config.latent_dim)
        self._logvar = nn.Linear(encoder_output_channels * 7 * 7, self._config.latent_dim)

        # [B,D] -> [B,W,H,3]
        self._decoder = Decoder(config)

    def forward(self, x):
        # grab some 2d features
        features = self._encoder(x)
        if nan_check(features): print("features have nan")
        
        # loses structure and channels - do not do
        # flatten 3d feature map into 1d per sample in batch
        # features = torch.flatten(features, start_dim=1)

        # good approach
        # features = self._gap(features).flatten(1)
        
        # overfit test approach
        features = features.flatten(1)

        mu = self._mu(features)  # position in latent space
        lv = self._logvar(features)  # confidence of position in latent space
        clipped_lv = torch.clamp(lv, min=-10.0, max=-2.0)  # help combat exploding gradients
        std = torch.exp(clipped_lv / 2.0)
        sample = mu  + (torch.randn_like(std) * std)  # vary it for automatic encoding
        if nan_check(sample): print("samples have nan")

        x_hat = self._decoder(sample)
        return x_hat, mu, clipped_lv


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
