"""
MDN-RNN training loop on pre-encoded latent sequences.

Key: gradient clipping is CRITICAL. MDN loss involves log/exp that can
produce extreme gradients. NaN detection with automatic skip.

Usage:
  python src/training/train_mdn_rnn.py
  python src/training/train_mdn_rnn.py --epochs 50 --seq_len 64
"""

import argparse
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.dataset import LatentSequenceDataset
from src.models.mdn_rnn import MDNRNN, mdn_loss
from src.utils.logging_setup import setup_logger
from src.utils.reproducibility import set_seed

logger = setup_logger("train_mdn_rnn")


def train_one_epoch(model, dataloader, optimizer, device, grad_clip):
    model.train()
    loss_sum = 0.0
    num_batches = 0

    for batch in dataloader:
        z_seq = batch["z"].to(device)
        actions = batch["actions"].to(device)

        z_input = z_seq[:, :-1, :]
        z_target = z_seq[:, 1:, :]

        pi, mu, log_sigma, _ = model(z_input, actions)
        loss = mdn_loss(z_target, pi, mu, log_sigma)

        if torch.isnan(loss):
            logger.warning("NaN loss detected! Skipping batch.")
            continue

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        loss_sum += loss.item()
        num_batches += 1

    if num_batches == 0:
        return {"loss": float("nan")}
    return {"loss": loss_sum / num_batches}


@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    loss_sum = 0.0
    num_batches = 0

    for batch in dataloader:
        z_seq = batch["z"].to(device)
        actions = batch["actions"].to(device)

        z_input = z_seq[:, :-1, :]
        z_target = z_seq[:, 1:, :]

        pi, mu, log_sigma, _ = model(z_input, actions)
        loss = mdn_loss(z_target, pi, mu, log_sigma)

        if not torch.isnan(loss):
            loss_sum += loss.item()
            num_batches += 1

    if num_batches == 0:
        return {"loss": float("nan")}
    return {"loss": loss_sum / num_batches}


def train(
    data_dir="data/carracing/encoded",
    epochs=50,
    batch_size=64,
    seq_len=32,
    learning_rate=1e-3,
    latent_dim=32,
    action_dim=3,
    hidden_dim=256,
    num_gaussians=5,
    grad_clip=1.0,
    save_dir="checkpoints/mdn_rnn",
    save_every=10,
    seed=42,
    num_workers=4,
    val_split=0.1,
):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    dataset = LatentSequenceDataset(data_dir, seq_len=seq_len)

    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    logger.info(f"Train: {train_size:,} sequences, Val: {val_size:,} sequences")

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    model = MDNRNN(
        latent_dim=latent_dim, action_dim=action_dim,
        hidden_dim=hidden_dim, num_gaussians=num_gaussians,
    ).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"MDN-RNN parameters: {total_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir="runs/mdn_rnn")

    best_val_loss = float("inf")
    nan_streak = 0
    MAX_NAN_STREAK = 5

    logger.info(
        f"Starting training: {epochs} epochs, batch_size={batch_size}, "
        f"seq_len={seq_len}, lr={learning_rate}, grad_clip={grad_clip}"
    )

    for epoch in range(epochs):
        epoch_start = time.time()

        train_metrics = train_one_epoch(model, train_loader, optimizer, device, grad_clip)
        val_metrics = evaluate(model, val_loader, device)
        epoch_time = time.time() - epoch_start

        if torch.isnan(torch.tensor(train_metrics["loss"])):
            nan_streak += 1
            logger.warning(f"Epoch {epoch + 1}: NaN loss (streak: {nan_streak}/{MAX_NAN_STREAK})")
            if nan_streak >= MAX_NAN_STREAK:
                logger.error("Too many NaN epochs. Stopping. Try lower lr or higher grad_clip.")
                break
            continue
        else:
            nan_streak = 0

        writer.add_scalar("train/loss", train_metrics["loss"], epoch)
        writer.add_scalar("val/loss", val_metrics["loss"], epoch)

        logger.info(
            f"Epoch {epoch + 1:3d}/{epochs} | "
            f"train_loss={train_metrics['loss']:8.2f} | "
            f"val_loss={val_metrics['loss']:8.2f} | "
            f"{epoch_time:.1f}s"
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": best_val_loss,
                    "config": {
                        "latent_dim": latent_dim, "action_dim": action_dim,
                        "hidden_dim": hidden_dim, "num_gaussians": num_gaussians,
                        "seq_len": seq_len, "epochs": epochs,
                        "batch_size": batch_size, "learning_rate": learning_rate,
                        "grad_clip": grad_clip,
                    },
                },
                save_path / "best.pt",
            )
            logger.info(f"  -> New best model saved (val_loss={best_val_loss:.2f})")

        if (epoch + 1) % save_every == 0:
            torch.save(
                {"epoch": epoch, "model_state_dict": model.state_dict(),
                 "optimizer_state_dict": optimizer.state_dict(),
                 "val_loss": val_metrics["loss"]},
                save_path / f"epoch_{epoch + 1:03d}.pt",
            )

    torch.save(
        {"epoch": epochs - 1, "model_state_dict": model.state_dict(),
         "config": {"latent_dim": latent_dim, "action_dim": action_dim,
                    "hidden_dim": hidden_dim, "num_gaussians": num_gaussians}},
        save_path / "final.pt",
    )
    writer.close()
    logger.info(f"Training complete. Best val_loss: {best_val_loss:.2f}")
    logger.info(f"Checkpoints saved to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MDN-RNN on encoded latent sequences.")
    parser.add_argument("--data_dir", type=str, default="data/carracing/encoded")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seq_len", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--latent_dim", type=int, default=32)
    parser.add_argument("--action_dim", type=int, default=3)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_gaussians", type=int, default=5)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--save_dir", type=str, default="checkpoints/mdn_rnn")
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    train(
        data_dir=args.data_dir, epochs=args.epochs, batch_size=args.batch_size,
        seq_len=args.seq_len, learning_rate=args.learning_rate,
        latent_dim=args.latent_dim, action_dim=args.action_dim,
        hidden_dim=args.hidden_dim, num_gaussians=args.num_gaussians,
        grad_clip=args.grad_clip, save_dir=args.save_dir,
        save_every=args.save_every, seed=args.seed, num_workers=args.num_workers,
    )
