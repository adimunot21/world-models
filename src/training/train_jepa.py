"""
JEPA training loop for CarRacing. End-to-end encoder + predictor.
Loss: prediction MSE + SIGReg.

Usage:
  python src/training/train_jepa.py
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.models.jepa import CarRacingJEPA, jepa_loss
from src.utils.logging_setup import setup_logger
from src.utils.reproducibility import set_seed

logger = setup_logger("train_jepa")


class JEPAFrameSequenceDataset(Dataset):
    """Raw frame sequences for JEPA training. No pre-encoding needed."""
    def __init__(self, data_dir="data/carracing", seq_len=8, max_rollouts=None):
        self.seq_len = seq_len
        self.data_dir = Path(data_dir)
        rollout_files = sorted(self.data_dir.glob("rollout_*.npz"))
        if not rollout_files:
            raise FileNotFoundError(f"No rollout files in {self.data_dir}")
        if max_rollouts is not None:
            rollout_files = rollout_files[:max_rollouts]

        logger.info(f"Loading {len(rollout_files)} rollouts for JEPA training...")
        self.observations = []
        self.actions = []
        self.index_map = []

        for i, rpath in enumerate(rollout_files):
            data = np.load(rpath)
            self.observations.append(data["observations"])
            self.actions.append(data["actions"])
            ep_len = len(data["observations"])
            for start in range(max(0, ep_len - seq_len)):
                self.index_map.append((i, start))

        logger.info(f"Loaded {len(self.observations)} rollouts, "
                     f"{len(self.index_map):,} sequences of length {seq_len}")

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        rollout_idx, start = self.index_map[idx]
        end = start + self.seq_len + 1
        obs = self.observations[rollout_idx][start:end].astype(np.float32) / 255.0
        obs = np.transpose(obs, (0, 3, 1, 2))
        actions = self.actions[rollout_idx][start:start + self.seq_len]
        return {
            "frames": torch.from_numpy(obs),
            "actions": torch.from_numpy(actions.astype(np.float32)),
        }


def train_one_epoch(model, dataloader, optimizer, device, sigreg_weight):
    model.train()
    total_sum = pred_sum = sig_sum = 0.0
    num_batches = 0
    for batch in dataloader:
        frames = batch["frames"].to(device)
        actions = batch["actions"].to(device)
        output = model(frames, actions)
        loss, pred_loss, sig_loss = jepa_loss(output, model.sigreg, sigreg_weight)
        if torch.isnan(loss):
            logger.warning("NaN loss, skipping batch")
            continue
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_sum += loss.item()
        pred_sum += pred_loss.item()
        sig_sum += sig_loss.item()
        num_batches += 1
    if num_batches == 0:
        return {"total": float("nan"), "pred": float("nan"), "sig": float("nan")}
    return {"total": total_sum / num_batches, "pred": pred_sum / num_batches, "sig": sig_sum / num_batches}


@torch.no_grad()
def evaluate(model, dataloader, device, sigreg_weight):
    model.eval()
    total_sum = pred_sum = sig_sum = 0.0
    num_batches = 0
    for batch in dataloader:
        frames = batch["frames"].to(device)
        actions = batch["actions"].to(device)
        output = model(frames, actions)
        loss, pred_loss, sig_loss = jepa_loss(output, model.sigreg, sigreg_weight)
        if not torch.isnan(loss):
            total_sum += loss.item()
            pred_sum += pred_loss.item()
            sig_sum += sig_loss.item()
            num_batches += 1
    if num_batches == 0:
        return {"total": float("nan"), "pred": float("nan"), "sig": float("nan")}
    return {"total": total_sum / num_batches, "pred": pred_sum / num_batches, "sig": sig_sum / num_batches}


def train(data_dir="data/carracing", epochs=50, batch_size=64, seq_len=8,
          learning_rate=1e-3, embedding_dim=192, sigreg_weight=1.0,
          save_dir="checkpoints/jepa", save_every=10, seed=42,
          num_workers=4, val_split=0.1, max_rollouts=None):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    dataset = JEPAFrameSequenceDataset(data_dir, seq_len=seq_len, max_rollouts=max_rollouts)
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    logger.info(f"Train: {train_size:,}, Val: {val_size:,}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)

    model = CarRacingJEPA(embedding_dim=embedding_dim, action_dim=3).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"JEPA parameters: {total_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir="runs/jepa_carracing")

    best_val_loss = float("inf")
    logger.info(f"Starting JEPA training: {epochs} epochs, batch_size={batch_size}, "
                f"seq_len={seq_len}, lr={learning_rate}, sigreg_weight={sigreg_weight}")

    for epoch in range(epochs):
        epoch_start = time.time()
        train_m = train_one_epoch(model, train_loader, optimizer, device, sigreg_weight)
        val_m = evaluate(model, val_loader, device, sigreg_weight)
        epoch_time = time.time() - epoch_start

        writer.add_scalar("train/total_loss", train_m["total"], epoch)
        writer.add_scalar("train/pred_loss", train_m["pred"], epoch)
        writer.add_scalar("train/sig_loss", train_m["sig"], epoch)
        writer.add_scalar("val/total_loss", val_m["total"], epoch)

        logger.info(
            f"Epoch {epoch + 1:3d}/{epochs} | "
            f"train={train_m['total']:7.4f} (pred={train_m['pred']:.4f}, sig={train_m['sig']:.4f}) | "
            f"val={val_m['total']:7.4f} | {epoch_time:.1f}s"
        )

        if val_m["total"] < best_val_loss:
            best_val_loss = val_m["total"]
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": best_val_loss,
                "config": {"embedding_dim": embedding_dim, "action_dim": 3,
                           "seq_len": seq_len, "sigreg_weight": sigreg_weight,
                           "epochs": epochs, "batch_size": batch_size,
                           "learning_rate": learning_rate},
            }, save_path / "best.pt")
            logger.info(f"  -> New best (val={best_val_loss:.4f})")

        if (epoch + 1) % save_every == 0:
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                         "val_loss": val_m["total"]}, save_path / f"epoch_{epoch + 1:03d}.pt")

    torch.save({"epoch": epochs - 1, "model_state_dict": model.state_dict(),
                "config": {"embedding_dim": embedding_dim, "action_dim": 3}},
               save_path / "final.pt")
    writer.close()
    logger.info(f"Training complete. Best val={best_val_loss:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train JEPA on CarRacing.")
    parser.add_argument("--data_dir", type=str, default="data/carracing")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seq_len", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--embedding_dim", type=int, default=192)
    parser.add_argument("--sigreg_weight", type=float, default=1.0)
    parser.add_argument("--save_dir", type=str, default="checkpoints/jepa")
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_rollouts", type=int, default=None)
    args = parser.parse_args()
    train(**vars(args))
