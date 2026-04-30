"""
Convolutional Variational Autoencoder (VAE) for CarRacing frames.

Architecture follows Ha & Schmidhuber 2018: CNN encoder compresses
64x64x3 RGB frames into a 32-dim latent vector z, transposed-CNN
decoder reconstructs from z.

What a VAE does vs a regular autoencoder:
  A regular autoencoder learns deterministic x -> z -> x'.
  A VAE learns a distribution over z: encoder outputs mu and sigma,
  z is sampled from N(mu, sigma^2). KL divergence pushes this toward
  N(0, I), ensuring the latent space is smooth and continuous.

The reparameterization trick:
  Can't backprop through sampling. Instead of z ~ N(mu, sigma^2),
  sample eps ~ N(0, 1) and compute z = mu + sigma * eps. Randomness
  is external to the computation graph, gradients flow through mu/sigma.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    """CNN encoder: (B, 3, 64, 64) -> mu (B, latent_dim), log_var (B, latent_dim).

    4 conv layers with stride 2, each halving spatial dims: 64->32->16->8->4.
    Then flatten and project to mu and log_var.

    Why log_var instead of sigma directly?
      log_var is unconstrained (any real number). Sigma must be positive.
      We recover sigma via exp(0.5 * log_var).
    """

    def __init__(self, img_channels: int = 3, latent_dim: int = 32):
        super().__init__()
        self.conv1 = nn.Conv2d(img_channels, 32, 4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(128, 256, 4, stride=2, padding=1)
        self.fc_mu = nn.Linear(256 * 4 * 4, latent_dim)
        self.fc_log_var = nn.Linear(256 * 4 * 4, latent_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(h))
        h = h.reshape(h.size(0), -1)
        return self.fc_mu(h), self.fc_log_var(h)


class Decoder(nn.Module):
    """Transposed-CNN decoder: (B, latent_dim) -> (B, 3, 64, 64).

    Mirrors encoder: project z to (256,4,4), then 4 deconv layers
    upsample 4->8->16->32->64. Final sigmoid maps to [0,1].
    """

    def __init__(self, img_channels: int = 3, latent_dim: int = 32):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 256 * 4 * 4)
        self.deconv1 = nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1)
        self.deconv4 = nn.ConvTranspose2d(32, img_channels, 4, stride=2, padding=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.fc(z))
        h = h.reshape(-1, 256, 4, 4)
        h = F.relu(self.deconv1(h))
        h = F.relu(self.deconv2(h))
        h = F.relu(self.deconv3(h))
        return torch.sigmoid(self.deconv4(h))


class VAE(nn.Module):
    """Full VAE: encoder + reparameterization + decoder.

    This is the 'V' (Vision) component of the world model.
    """

    def __init__(self, img_channels: int = 3, latent_dim: int = 32):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = Encoder(img_channels, latent_dim)
        self.decoder = Decoder(img_channels, latent_dim)

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """z = mu + sigma * eps, where eps ~ N(0, I)."""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + std * eps

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decoder(z)
        return x_recon, mu, log_var, z

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (mu, log_var) without sampling or decoding."""
        return self.encoder(x)

    def encode_to_z(self, x: torch.Tensor) -> torch.Tensor:
        """Encode and sample z. Convenience for inference."""
        mu, log_var = self.encoder(x)
        return self.reparameterize(mu, log_var)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Reconstruct image from latent vector."""
        return self.decoder(z)


def vae_loss(
    x: torch.Tensor,
    x_recon: torch.Tensor,
    mu: torch.Tensor,
    log_var: torch.Tensor,
    kl_weight: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """VAE loss = reconstruction (BCE) + kl_weight * KL divergence.

    BCE per pixel since images are [0,1]. Sum over pixels, avg over batch.
    KL(N(mu, sigma^2) || N(0,1)) = -0.5 * sum(1 + log_var - mu^2 - exp(log_var)).
    """
    recon_loss = F.binary_cross_entropy(x_recon, x, reduction="sum") / x.size(0)
    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) / x.size(0)
    total_loss = recon_loss + kl_weight * kl_loss
    return total_loss, recon_loss, kl_loss


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VAE(img_channels=3, latent_dim=32).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    enc_params = sum(p.numel() for p in model.encoder.parameters())
    dec_params = sum(p.numel() for p in model.decoder.parameters())
    print(f"Total params:   {total_params:,}")
    print(f"Encoder params: {enc_params:,}")
    print(f"Decoder params: {dec_params:,}")

    x = torch.randn(4, 3, 64, 64, device=device).clamp(0, 1)
    x_recon, mu, log_var, z = model(x)
    print(f"\nInput:   {x.shape}")
    print(f"Recon:   {x_recon.shape}, range=[{x_recon.min():.3f}, {x_recon.max():.3f}]")
    print(f"mu:      {mu.shape}")
    print(f"log_var: {log_var.shape}")
    print(f"z:       {z.shape}")

    loss, recon, kl = vae_loss(x, x_recon, mu, log_var, kl_weight=1.0)
    print(f"\nLoss: {loss.item():.1f} (recon={recon.item():.1f}, kl={kl.item():.1f})")

    mem_mb = torch.cuda.max_memory_allocated() / 1e6
    print(f"Peak VRAM: {mem_mb:.0f} MB")
