"""
define models used in project
"""
import torch
import torch.nn as nn

class VAEConfig():
    def __init__(self, latent_dim=256, cbld=None):
        self.x_w = 224
        self.x_h = 224
        self.feature_w = 7
        self.feature_h = 7
        self.latent_dim = latent_dim
        self.p_s = 32
        self.core_block_lin_dim = cbld


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

"""
Basic ResNet block (like ResNet-18/34):
  conv3x3 -> BN -> ReLU -> conv3x3 -> BN
Skip:
  identity if (ic==oc and stride==1) else 1x1 conv with stride
"""
class ResBlock(nn.Module):
    def __init__(self, ic, oc, stride=1):
        super().__init__()
        self._main = nn.Sequential(
            nn.Conv2d(ic, oc, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(oc),
            nn.ReLU(inplace=True),
            nn.Conv2d(oc, oc, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(oc),
        )

        self.down = None
        if stride != 1 or ic != oc:
            self.down = nn.Sequential(
                nn.Conv2d(ic, oc, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(oc),
            )

    def forward(self, x):
        identity = x
        out = self._main(x)

        if self.down is not None:
            identity = self.down(identity)

        out = out + identity
        out = nn.ReLU(inplace=True)(out)
        return out


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


class ResBlock_NoBN(nn.Module):
    def __init__(self, c, act=nn.ReLU(inplace=True)):
        super().__init__()
        self.conv1 = nn.Conv2d(c, c, 3, padding=1)
        self.act = act
        self.conv2 = nn.Conv2d(c, c, 3, padding=1)

    def forward(self, x):
        return x + self.conv2(self.act(self.conv1(x)))


class ResUpBlock(nn.Module):
    def __init__(self, ic, oc, act=nn.ReLU(inplace=True)):
        super().__init__()
        self._up = nn.Upsample(scale_factor=2, mode="nearest")
        self._conv = nn.Conv2d(ic, oc, kernel_size=3, padding=1)
        self._res = ResBlock_NoBN(oc, act)
        self._skip = (
            nn.Conv2d(ic, oc, 1)
            if ic != oc
            else nn.Identity()
        )

    def forward(self, x):
        # upsample before sending through residual
        x_upped = self._up(x)

        # run upsampled x through normal conv,
        # identity gets rechanneled if needed
        y = self._conv(x_upped)
        skip = self._skip(x_upped)

        # combine and run through simple conv-act-conv res
        return self._res(y + skip)


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


"""
ResNet-ish replacement
Output spatial size: 7x7 (for 224x224 input), same as before.
"""
class ResEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self._config = config

        self._encoder = nn.Sequential(
            # 224x224
            ResBlock(3, 64, stride=2),
            ResBlock(64, 64, stride=1),

            # 112x112
            ResBlock(64, 128, stride=2),
            ResBlock(128, 128, stride=1),

            # 56x56
            ResBlock(128, 192, stride=2),
            ResBlock(192, 192, stride=1),

            # 28x28
            ResBlock(192, 256, stride=2),
            ResBlock(256, 256, stride=1),

            # 14x14
            ResBlock(256, 256, stride=2),
            # ResBlock(256, 256, stride=1),
            # -> 7x7
        )

    def forward(self, x):
        return self._encoder(x)


"""
ViT encoder
"""
class ViTBlock(nn.Module):
    def __init__(self, config, n_heads=12, dim=768, lin_dim=3072):
        super().__init__()
        self._config = config
        lin_dim = config.core_block_lin_dim or lin_dim
        self._ln = nn.LayerNorm(config.latent_dim)
        self._mhsa = nn.MultiheadAttention(config.latent_dim, n_heads, batch_first=True)
        self._ln2 = nn.LayerNorm(config.latent_dim)
        self._mlp = nn.Sequential(
            nn.Linear(config.latent_dim, lin_dim),
            nn.GELU(),
            nn.Linear(lin_dim, config.latent_dim),
        )

    def forward(self, image_tokens):
        # attention half
        it = self._ln(image_tokens)
        at, _ = self._mhsa(it, it, it, need_weights=False)
        it = image_tokens + at
        # linear half
        return it + self._mlp(self._ln2(it))


class ViTVAEAdapter(nn.Module):
    def __init__(self, config, n_heads=12, dim=768, lin_dim=3072, out_dim=16):
        super().__init__()
        self._config = config
        self._dim = dim
        self._ln_q = nn.LayerNorm(dim)
        self._ln_kv = nn.LayerNorm(dim)
        self._xattn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self._to_sl = nn.Conv2d(dim, out_dim, kernel_size=1)

    def forward(self, patch, cls):
        # patch: [B,N,D], cls: [B,1,D]
        # attention half (each patch queries global)
        q = self._ln_q(patch)
        kv = self._ln_kv(cls)
        a, _ = self._xattn(q, kv, kv, need_weights=False)
        patch = patch + a
        
        grid = patch.transpose(1, 2).reshape(
            patch.shape[0], 
            self._dim, 
            self._config.feature_w, 
            self._config.feature_h,
        )
        return self._to_sl(grid)


class ViTEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self._conv_as_embed = nn.Conv2d(
            3,  # assume RGB always 
            config.latent_dim, 
            kernel_size=config.p_s, 
            stride=config.p_s,
        )
        self._cls = nn.Parameter(torch.zeros(1, 1, config.latent_dim))

        # L layers of blocks with N heads and D dim
        self._core_blocks = nn.Sequential(
            *[ViTBlock(config, n_heads=4) for _ in range(5)]
        )
        
        # Spatial latent transform
        self._vva = ViTVAEAdapter(config, n_heads=4, dim=config.latent_dim, lin_dim=256, out_dim=256)

    def forward(self, x):
        # tokenize image, add CLS token
        image_tokens = self._conv_as_embed(x)
        all_tokens = torch.cat([
            self._cls.expand(x.shape[0], -1, -1), 
            image_tokens.flatten(2).transpose(1,2),
        ], dim=1)

        t = self._core_blocks(all_tokens)

        # have each position token attend to each global class token
        cls = t[:, :1, :]
        pos = t[:, 1:, :]
        
        return self._vva(pos, cls)


"""
Base decoder
"""
class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self._config = config

        # latent comes with spatial structure - no need to recreate
        # seed does not upsample, but preps signal for upsampling
        self._seed = nn.Sequential(
            nn.Conv2d(16, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        
        # focused on upsampling into a real RGB image
        self._decoder = nn.Sequential(
            ConvUpBlock_NoBN(192, 128, act=nn.Identity()),  # 7
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            ConvUpBlock_NoBN(128, 96),   # 14
            nn.Conv2d(96, 64, kernel_size=3, padding=1),
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

"""
Apply resnet principles to decoding in an attempt
to eliminate the need to allocate KL on stupid stuff
"""
class ResDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self._config = config
        
        # same meta-arch as v1 decoder - upsample channels before upsampling space
        # repeat upchannel -> act -> res pattern
        self._seed = nn.Sequential(
            nn.Conv2d(16, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            ResBlock_NoBN(64),
            # nn.Conv2d(32, 64, 3, padding=1),
            # nn.ReLU(inplace=True),
            # ResBlock_NoBN(64),
            # nn.Conv2d(64, 128, 3, padding=1),
            # nn.ReLU(inplace=True),
            # ResBlock_NoBN(128),
            # nn.Conv2d(96, 128, 3, padding=1),
            # nn.ReLU(inplace=True),
            # ResBlock_NoBN(128),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            ResBlock_NoBN(128),
        )

        self._decode = nn.Sequential(
            ResUpBlock(128, 96),  # 7x7 -> 14x14
            # ResBlock_NoBN(128),
            ResUpBlock(96, 64),  # 14 -> 28
            # ResBlock_NoBN(96),
            ResUpBlock(64, 64),  # 28 -> 56
            # ResBlock_NoBN(64),
            ResUpBlock(64, 32),  # 56 -> 112
            # ResBlock_NoBN(32),
            ResUpBlock(32, 16),  # 112 -> 224
            # ResBlock_NoBN(16),
            nn.Conv2d(16, 3, kernel_size=3, padding=1),
        )

    def forward(self, z):
        h = self._seed(z)
        return self._decode(h)


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
Full x -> x' models 
"""
class ViTVAE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self._config = config
        self._encoder = ViTEncoder(config)
        self._post_head = SpatialPosteriorHead(256, 16)
        self._decoder = ResDecoder(config)

    def forward(self, x):
        f = self._encoder(x)
        mu, lv = self._post_head(f)
        clipped_lv = torch.clamp(lv, min=-10.0, max=1.0)  # help combat exploding gradients
        std = torch.exp(clipped_lv / 2.0)
        sample = mu  + (torch.randn_like(std) * std)  # reparam trick

        x_hat = self._decoder(sample)
        return x_hat, sample, mu, clipped_lv


class VariationalAutoEncoderRes(nn.Module):
    def __init__(self, config):
        super().__init__()
        self._config = config

        # [B,3,W,H] -> [B,C,7,7]
        self._encoder = ResEncoder(config)

        # Learned (spatial) posterior weights
        # [B,C,7,7] -> [B,D,7,7]
        self._post_head = SpatialPosteriorHead(256, 16)

        # [B,C,7,7] -> [B,3,W,H]
        self._decoder = ResDecoder(config)

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
