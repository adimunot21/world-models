"""
VAE reconstruction visualizations.

Generates three plots:
  1. Original vs Reconstructed frames (side-by-side grid)
  2. Latent space traversal (vary one z dim, see visual effect)
  3. Random samples from prior z ~ N(0, 1)

Usage:
  python src/visualization/reconstruction_viz.py --checkpoint checkpoints/vae/best.pt
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.dataset import FrameDataset
from src.models.vae import VAE
from src.utils.logging_setup import setup_logger

logger = setup_logger("vae_viz")


def load_vae(checkpoint_path: str, device: torch.device) -> VAE:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    latent_dim = ckpt.get("latent_dim", 32)
    model = VAE(img_channels=3, latent_dim=latent_dim).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    logger.info(f"Loaded VAE (latent_dim={latent_dim}) from {checkpoint_path}")
    return model


def tensor_to_img(t: torch.Tensor) -> np.ndarray:
    img = t.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    return (img * 255).astype(np.uint8)


@torch.no_grad()
def plot_reconstructions(model, dataset, device, output_path, num_samples=8):
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    frames = torch.stack([dataset[i] for i in indices]).to(device)
    x_recon, mu, log_var, z = model(frames)

    fig, axes = plt.subplots(2, num_samples, figsize=(2.5 * num_samples, 5))
    fig.suptitle("VAE Reconstructions: Original (top) vs Reconstructed (bottom)", fontsize=13)

    for i in range(num_samples):
        axes[0, i].imshow(tensor_to_img(frames[i]))
        axes[0, i].axis("off")
        if i == 0:
            axes[0, i].set_ylabel("Original", fontsize=11)
        axes[1, i].imshow(tensor_to_img(x_recon[i]))
        axes[1, i].axis("off")
        if i == 0:
            axes[1, i].set_ylabel("Recon", fontsize=11)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Reconstruction grid saved to {output_path}")


@torch.no_grad()
def plot_latent_traversal(model, dataset, device, output_path, num_dims=8, num_steps=7):
    ref_frame = dataset[np.random.randint(len(dataset))].unsqueeze(0).to(device)
    mu, log_var = model.encode(ref_frame)
    z_ref = mu.squeeze(0)

    fig, axes = plt.subplots(num_dims, num_steps, figsize=(2 * num_steps, 2 * num_dims))
    fig.suptitle("Latent Traversal: Each row varies one z dimension from -3 to +3", fontsize=13)

    for dim in range(num_dims):
        values = torch.linspace(-3, 3, num_steps)
        for step, val in enumerate(values):
            z = z_ref.clone()
            z[dim] = val
            decoded = model.decode(z.unsqueeze(0))
            axes[dim, step].imshow(tensor_to_img(decoded.squeeze(0)))
            axes[dim, step].axis("off")
            if step == 0:
                axes[dim, step].set_ylabel(f"z[{dim}]", fontsize=9)
            if dim == 0:
                axes[dim, step].set_title(f"{val:.1f}", fontsize=9)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close()
    logger.info(f"Latent traversal saved to {output_path}")


@torch.no_grad()
def plot_random_samples(model, device, output_path, num_samples=16):
    z = torch.randn(num_samples, model.latent_dim, device=device)
    decoded = model.decode(z)

    cols = 4
    rows = (num_samples + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(2.5 * cols, 2.5 * rows))
    fig.suptitle("Random Samples from Prior z ~ N(0, 1)", fontsize=13)

    for i in range(num_samples):
        r, c = divmod(i, cols)
        ax = axes[r, c] if rows > 1 else axes[c]
        ax.imshow(tensor_to_img(decoded[i]))
        ax.axis("off")
    for i in range(num_samples, rows * cols):
        r, c = divmod(i, cols)
        ax = axes[r, c] if rows > 1 else axes[c]
        ax.axis("off")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Random samples saved to {output_path}")


def main(checkpoint="checkpoints/vae/best.pt", data_dir="data/carracing", output_dir="results/plots"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_vae(checkpoint, device)
    dataset = FrameDataset(data_dir, max_rollouts=10)

    out = Path(output_dir)
    plot_reconstructions(model, dataset, device, out / "vae_reconstructions.png")
    plot_latent_traversal(model, dataset, device, out / "vae_latent_traversal.png")
    plot_random_samples(model, device, out / "vae_random_samples.png")
    logger.info("All VAE visualizations complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate VAE visualizations.")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/vae/best.pt")
    parser.add_argument("--data_dir", type=str, default="data/carracing")
    parser.add_argument("--output_dir", type=str, default="results/plots")
    args = parser.parse_args()
    main(checkpoint=args.checkpoint, data_dir=args.data_dir, output_dir=args.output_dir)
