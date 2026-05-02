"""
Head-to-head comparison of VAE vs JEPA representations on CarRacing.
Linear probing, PCA visualization, embedding distribution analysis.

Usage:
  python src/evaluation/compare.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.models.vae import VAE
from src.models.jepa import CarRacingJEPA
from src.utils.logging_setup import setup_logger

logger = setup_logger("compare")


def load_vae(checkpoint, device):
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    latent_dim = ckpt.get("latent_dim", 32)
    model = VAE(img_channels=3, latent_dim=latent_dim).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def load_jepa(checkpoint, device):
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})
    model = CarRacingJEPA(
        embedding_dim=cfg.get("embedding_dim", 192),
        action_dim=cfg.get("action_dim", 3),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


@torch.no_grad()
def extract_representations(vae, jepa, data_dir, device, max_rollouts=50, samples_per_rollout=20):
    data_path = Path(data_dir)
    rollout_files = sorted(data_path.glob("rollout_*.npz"))[:max_rollouts]

    all_frames, all_vae_z, all_jepa_emb, all_rewards, all_actions = [], [], [], [], []

    for rpath in rollout_files:
        data = np.load(rpath)
        obs, rewards, actions = data["observations"], data["rewards"], data["actions"]
        indices = np.random.choice(len(obs), min(samples_per_rollout, len(obs)), replace=False)

        all_frames.append(obs[indices])
        all_rewards.append(rewards[indices])
        all_actions.append(actions[indices])

        frames_tensor = torch.from_numpy(obs[indices].astype(np.float32) / 255.0)
        frames_tensor = frames_tensor.permute(0, 3, 1, 2).to(device)

        mu, _ = vae.encode(frames_tensor)
        all_vae_z.append(mu.cpu().numpy())

        emb = jepa.encode(frames_tensor)
        all_jepa_emb.append(emb.cpu().numpy())

    return {
        "frames": np.concatenate(all_frames),
        "vae_z": np.concatenate(all_vae_z),
        "jepa_emb": np.concatenate(all_jepa_emb),
        "rewards": np.concatenate(all_rewards),
        "actions": np.concatenate(all_actions),
    }


def probe_representations(reps, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for name, embeddings in [("VAE (32d)", reps["vae_z"]), ("JEPA (192d)", reps["jepa_emb"])]:
        n = len(embeddings)
        split = int(0.8 * n)
        X_train, X_test = embeddings[:split], embeddings[split:]
        y_train, y_test = reps["rewards"][:split], reps["rewards"][split:]

        probe = Ridge(alpha=1.0)
        probe.fit(X_train, y_train)
        y_pred = probe.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        results[name] = {"reward_r2": r2}
        logger.info(f"{name}: reward probe R2 = {r2:.4f}")
    return results


def plot_latent_comparison(reps, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, (name, emb) in zip(axes, [("VAE (32d)", reps["vae_z"]), ("JEPA (192d)", reps["jepa_emb"])]):
        pca = PCA(n_components=2)
        emb_2d = pca.fit_transform(emb)
        scatter = ax.scatter(emb_2d[:, 0], emb_2d[:, 1], c=reps["rewards"], cmap="RdYlGn", s=3, alpha=0.5)
        ax.set_title(f"{name}\nPCA var explained: {pca.explained_variance_ratio_.sum():.1%}")
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
        plt.colorbar(scatter, ax=ax, label="Reward")
    fig.suptitle("Latent Space Comparison: VAE vs JEPA (colored by reward)", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / "latent_comparison_pca.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"PCA comparison saved")


def plot_embedding_distributions(reps, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, (name, emb) in zip(axes, [("VAE (32d)", reps["vae_z"]), ("JEPA (192d)", reps["jepa_emb"])]):
        for dim in range(min(8, emb.shape[1])):
            ax.hist(emb[:, dim], bins=50, alpha=0.4, label=f"dim {dim}")
        ax.set_title(f"{name}: First 8 dimensions")
        ax.set_xlabel("Value"); ax.set_ylabel("Count")
        ax.legend(fontsize=6, ncol=2)
    fig.suptitle("Embedding Distributions: VAE (KL) vs JEPA (SIGReg)", fontsize=13)
    plt.tight_layout()
    plt.savefig(output_dir / "embedding_distributions.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Distribution comparison saved")


def main(vae_checkpoint="checkpoints/vae/best.pt", jepa_checkpoint="checkpoints/jepa/best.pt",
         data_dir="data/carracing", output_dir="results/plots"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Loading models...")
    vae = load_vae(vae_checkpoint, device)
    jepa = load_jepa(jepa_checkpoint, device)

    logger.info("Extracting representations...")
    reps = extract_representations(vae, jepa, data_dir, device)
    logger.info(f"Extracted {len(reps['vae_z'])} samples")

    out = Path(output_dir)
    logger.info("Running linear probes...")
    probe_results = probe_representations(reps, out)

    logger.info("Generating visualizations...")
    plot_latent_comparison(reps, out)
    plot_embedding_distributions(reps, out)

    print("\n" + "=" * 60)
    print("PARADIGM COMPARISON: VAE vs JEPA on CarRacing")
    print("=" * 60)
    for name, metrics in probe_results.items():
        print(f"  {name}: reward R2 = {metrics['reward_r2']:.4f}")
    print(f"\nVAE: {reps['vae_z'].shape[1]}-dim, with decoder")
    print(f"JEPA: {reps['jepa_emb'].shape[1]}-dim, NO decoder")
    print("=" * 60)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--vae_checkpoint", default="checkpoints/vae/best.pt")
    parser.add_argument("--jepa_checkpoint", default="checkpoints/jepa/best.pt")
    parser.add_argument("--data_dir", default="data/carracing")
    parser.add_argument("--output_dir", default="results/plots")
    args = parser.parse_args()
    main(**vars(args))
