"""
Pre-encode the entire CarRacing dataset through the trained VAE.

Runs each frame once through the VAE encoder, saves latent vectors.
The MDN-RNN trains on these directly — no redundant encoding per epoch.

Usage:
  python src/training/encode_dataset.py
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.models.vae import VAE
from src.utils.logging_setup import setup_logger
from src.utils.reproducibility import set_seed

logger = setup_logger("encode")


@torch.no_grad()
def encode_dataset(
    checkpoint: str = "checkpoints/vae/best.pt",
    data_dir: str = "data/carracing",
    output_dir: str = "data/carracing/encoded",
    batch_size: int = 256,
    seed: int = 42,
) -> None:
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    latent_dim = ckpt.get("latent_dim", 32)
    vae = VAE(img_channels=3, latent_dim=latent_dim).to(device)
    vae.load_state_dict(ckpt["model_state_dict"])
    vae.eval()
    logger.info(f"Loaded VAE (latent_dim={latent_dim}) from {checkpoint}")

    data_path = Path(data_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    rollout_files = sorted(data_path.glob("rollout_*.npz"))
    if not rollout_files:
        logger.error(f"No rollout files in {data_path}")
        return

    existing = set(p.stem for p in out_path.glob("rollout_*.npz"))
    to_encode = [f for f in rollout_files if f.stem not in existing]

    if not to_encode:
        logger.info(f"All {len(rollout_files)} rollouts already encoded.")
        return

    logger.info(f"Encoding {len(to_encode)}/{len(rollout_files)} rollouts...")
    start_time = time.time()

    for i, rpath in enumerate(to_encode):
        data = np.load(rpath)
        obs = data["observations"]

        all_z = []
        all_mu = []

        for start in range(0, len(obs), batch_size):
            batch = obs[start : start + batch_size]
            frames = torch.from_numpy(batch.astype(np.float32) / 255.0)
            frames = frames.permute(0, 3, 1, 2).to(device)

            mu, log_var = vae.encode(frames)
            z = vae.reparameterize(mu, log_var)

            all_z.append(z.cpu().numpy())
            all_mu.append(mu.cpu().numpy())

        z_array = np.concatenate(all_z, axis=0)
        mu_array = np.concatenate(all_mu, axis=0)

        np.savez_compressed(
            out_path / rpath.name,
            z=z_array.astype(np.float32),
            mu=mu_array.astype(np.float32),
            actions=data["actions"],
            rewards=data["rewards"],
            dones=data["dones"],
        )

        if (i + 1) % 50 == 0 or i == 0:
            elapsed = time.time() - start_time
            speed = (i + 1) / elapsed
            eta = (len(to_encode) - i - 1) / max(speed, 1e-6)
            logger.info(
                f"Encoded {i + 1:4d}/{len(to_encode)} | "
                f"z shape={z_array.shape} | "
                f"speed={speed:.1f} rollouts/s | "
                f"ETA={eta:.0f}s"
            )

    elapsed_total = time.time() - start_time
    logger.info(f"Encoding complete. {len(to_encode)} rollouts in {elapsed_total:.0f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-encode dataset through VAE.")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/vae/best.pt")
    parser.add_argument("--data_dir", type=str, default="data/carracing")
    parser.add_argument("--output_dir", type=str, default="data/carracing/encoded")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    encode_dataset(
        checkpoint=args.checkpoint,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        seed=args.seed,
    )
