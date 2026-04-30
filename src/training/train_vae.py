"""
VAE training loop for CarRacing frames.

Implements KL warmup to prevent posterior collapse: start with kl_weight=0
(pure autoencoder), linearly increase to target over warmup epochs.

Usage:
  python src/training/train_vae.py
  python src/training/train_vae.py --epochs 50 --batch_size 64
"""

import argparse
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.dataset import FrameDataset
from src.models.vae import VAE, vae_loss
from src.utils.config import load_config
from src.utils.logging_setup import setup_logger
from src.utils.reproducibility import set_seed

logger = setup_logger("train_vae")


def train_one_epoch(model, dataloader, optimizer, device, kl_weight):
    model.train()
    total_loss_sum = recon_loss_sum = kl_loss_sum = 0.0
    num_batches = 0

    for batch in dataloader:
        x = batch.to(device)
        x_recon, mu, log_var, z = model(x)
        loss, recon, kl = vae_loss(x, x_recon, mu, log_var, kl_weight=kl_weight)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss_sum += loss.item()
        recon_loss_sum += recon.item()
        kl_loss_sum += kl.item()
        num_batches += 1

    return {
        "total_loss": total_loss_sum / num_batches,
        "recon_loss": recon_loss_sum / num_batches,
        "kl_loss": kl_loss_sum / num_batches,
    }


@torch.no_grad()
def evaluate(model, dataloader, device, kl_weight):
    model.eval()
    total_loss_sum = recon_loss_sum = kl_loss_sum = 0.0
    num_batches = 0

    for batch in dataloader:
        x = batch.to(device)
        x_recon, mu, log_var, z = model(x)
        loss, recon, kl = vae_loss(x, x_recon, mu, log_var, kl_weight=kl_weight)

        total_loss_sum += loss.item()
        recon_loss_sum += recon.item()
        kl_loss_sum += kl.item()
        num_batches += 1

    return {
        "total_loss": total_loss_sum / num_batches,
        "recon_loss": recon_loss_sum / num_batches,
        "kl_loss": kl_loss_sum / num_batches,
    }


def get_kl_weight(epoch, warmup_epochs, target_weight):
    if warmup_epochs <= 0:
        return target_weight
    return min(1.0, epoch / warmup_epochs) * target_weight


def train(
    data_dir="data/carracing",
    epochs=30,
    batch_size=128,
    learning_rate=1e-3,
    latent_dim=32,
    kl_weight=1.0,
    kl_warmup_epochs=5,
    save_dir="checkpoints/vae",
    save_every=5,
    seed=42,
    num_workers=4,
    val_split=0.1,
):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # --- Data ---
    dataset = FrameDataset(data_dir)
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    logger.info(f"Train: {train_size:,} frames, Val: {val_size:,} frames")

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    # --- Model ---
    model = VAE(img_channels=3, latent_dim=latent_dim).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"VAE parameters: {total_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # --- Logging ---
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=f"runs/vae_{latent_dim}d")

    # --- Training ---
    best_val_loss = float("inf")
    logger.info(f"Starting training: {epochs} epochs, batch_size={batch_size}, lr={learning_rate}")
    logger.info(f"KL warmup: {kl_warmup_epochs} epochs -> target weight={kl_weight}")

    for epoch in range(epochs):
        epoch_start = time.time()
        current_kl_weight = get_kl_weight(epoch, kl_warmup_epochs, kl_weight)

        train_metrics = train_one_epoch(model, train_loader, optimizer, device, current_kl_weight)
        val_metrics = evaluate(model, val_loader, device, current_kl_weight)
        epoch_time = time.time() - epoch_start

        for key, val in train_metrics.items():
            writer.add_scalar(f"train/{key}", val, epoch)
        for key, val in val_metrics.items():
            writer.add_scalar(f"val/{key}", val, epoch)
        writer.add_scalar("train/kl_weight", current_kl_weight, epoch)

        logger.info(
            f"Epoch {epoch + 1:3d}/{epochs} | "
            f"train_loss={train_metrics['total_loss']:8.1f} "
            f"(recon={train_metrics['recon_loss']:7.1f}, kl={train_metrics['kl_loss']:6.1f}) | "
            f"val_loss={val_metrics['total_loss']:8.1f} | "
            f"kl_w={current_kl_weight:.2f} | "
            f"{epoch_time:.1f}s"
        )

        if val_metrics["total_loss"] < best_val_loss:
            best_val_loss = val_metrics["total_loss"]
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": best_val_loss,
                    "latent_dim": latent_dim,
                    "config": {
                        "epochs": epochs, "batch_size": batch_size,
                        "learning_rate": learning_rate, "kl_weight": kl_weight,
                        "kl_warmup_epochs": kl_warmup_epochs,
                    },
                },
                save_path / "best.pt",
            )
            logger.info(f"  -> New best model saved (val_loss={best_val_loss:.1f})")

        if (epoch + 1) % save_every == 0:
            torch.save(
                {"epoch": epoch, "model_state_dict": model.state_dict(),
                 "optimizer_state_dict": optimizer.state_dict(),
                 "val_loss": val_metrics["total_loss"], "latent_dim": latent_dim},
                save_path / f"epoch_{epoch + 1:03d}.pt",
            )

    torch.save(
        {"epoch": epochs - 1, "model_state_dict": model.state_dict(), "latent_dim": latent_dim},
        save_path / "final.pt",
    )
    writer.close()
    logger.info(f"Training complete. Best val_loss: {best_val_loss:.1f}")
    logger.info(f"Checkpoints saved to {save_path}")
    logger.info(f"TensorBoard logs: tensorboard --logdir runs/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the VAE on CarRacing frames.")
    parser.add_argument("--data_dir", type=str, default="data/carracing")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--latent_dim", type=int, default=32)
    parser.add_argument("--kl_weight", type=float, default=1.0)
    parser.add_argument("--kl_warmup_epochs", type=int, default=5)
    parser.add_argument("--save_dir", type=str, default="checkpoints/vae")
    parser.add_argument("--save_every", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    train(
        data_dir=args.data_dir, epochs=args.epochs, batch_size=args.batch_size,
        learning_rate=args.learning_rate, latent_dim=args.latent_dim,
        kl_weight=args.kl_weight, kl_warmup_epochs=args.kl_warmup_epochs,
        save_dir=args.save_dir, save_every=args.save_every,
        seed=args.seed, num_workers=args.num_workers,
    )
