"""
Dream Environment — replace real CarRacing with MDN-RNN imagination.

Dream steps are LSTM forward + sampling (~0.1s vs ~6s for real env).
Controller trains entirely in imagination, then transfers to reality.
"""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.models.controller import Controller
from src.models.mdn_rnn import MDNRNN
from src.utils.logging_setup import setup_logger

logger = setup_logger("dream_env")


class DreamEnv:
    """Environment running entirely in MDN-RNN imagination.

    Args:
        mdn_rnn: Trained MDN-RNN.
        initial_z_dataset: (N, latent_dim) real z vectors to seed dreams.
        device: Torch device.
        temperature: MDN-RNN sampling temperature.
        max_steps: Maximum dream length.
    """

    def __init__(self, mdn_rnn, initial_z_dataset, device, temperature=1.0, max_steps=300):
        self.mdn_rnn = mdn_rnn.eval()
        self.initial_z_dataset = initial_z_dataset
        self.device = device
        self.temperature = temperature
        self.max_steps = max_steps
        self.current_z = None
        self.hidden = None
        self.step_count = 0

    def reset(self):
        """Start a new dream from a random real initial state.
        Returns: (z, h) tensors."""
        idx = np.random.randint(len(self.initial_z_dataset))
        z_np = self.initial_z_dataset[idx]
        self.current_z = torch.from_numpy(z_np).float().unsqueeze(0).to(self.device)
        self.hidden = self.mdn_rnn.init_hidden(1, self.device)
        self.step_count = 0
        h = self.hidden[0][-1]
        return self.current_z, h

    @torch.no_grad()
    def step(self, action):
        """One dream step. Returns: (z_next, h, reward, done)."""
        pi, mu, log_sigma, self.hidden = self.mdn_rnn.predict_next(
            self.current_z, action, self.hidden
        )
        z_next = self.mdn_rnn.sample_next_z(pi, mu, log_sigma, self.temperature)

        action_np = action.squeeze(0).cpu().numpy()
        reward = action_np[1] - action_np[2] - abs(action_np[0]) * 0.1

        self.current_z = z_next
        self.step_count += 1
        done = self.step_count >= self.max_steps

        h = self.hidden[0][-1]
        return z_next, h, reward, done


def collect_initial_z(encoded_dir="data/carracing/encoded", num_rollouts=50, samples_per_rollout=10):
    """Collect pool of real z vectors to seed dream initial states."""
    encoded_path = Path(encoded_dir)
    rollout_files = sorted(encoded_path.glob("rollout_*.npz"))[:num_rollouts]

    all_z = []
    for rpath in rollout_files:
        data = np.load(rpath)
        z = data["z"]
        indices = np.random.choice(len(z), min(samples_per_rollout, len(z)), replace=False)
        all_z.append(z[indices])

    z_pool = np.concatenate(all_z, axis=0)
    logger.info(f"Collected {len(z_pool)} initial z vectors from {len(rollout_files)} rollouts")
    return z_pool


def evaluate_in_dream(controller, mdn_rnn, initial_z_pool, device,
                       controller_params, num_rollouts=3, max_steps=300, temperature=1.0):
    """Evaluate a controller inside dreams. Returns mean dream reward."""
    controller.set_flat_params(controller_params)
    controller.eval()

    dream_env = DreamEnv(mdn_rnn, initial_z_pool, device, temperature, max_steps)
    rewards = []

    for _ in range(num_rollouts):
        z, h = dream_env.reset()
        total_reward = 0.0

        for _ in range(max_steps):
            action = controller(z, h)
            z, h, reward, done = dream_env.step(action)
            total_reward += reward
            if done:
                break

        rewards.append(total_reward)

    return float(np.mean(rewards))


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rnn_ckpt = torch.load("checkpoints/mdn_rnn/best.pt", map_location=device, weights_only=False)
    rnn_config = rnn_ckpt.get("config", {})
    mdn_rnn = MDNRNN(
        latent_dim=rnn_config.get("latent_dim", 32),
        action_dim=rnn_config.get("action_dim", 3),
        hidden_dim=rnn_config.get("hidden_dim", 256),
        num_gaussians=rnn_config.get("num_gaussians", 5),
    ).to(device)
    mdn_rnn.load_state_dict(rnn_ckpt["model_state_dict"])
    mdn_rnn.eval()

    z_pool = collect_initial_z()

    dream_env = DreamEnv(mdn_rnn, z_pool, device, temperature=1.0, max_steps=100)
    z, h = dream_env.reset()
    print(f"Initial z: {z.shape}, h: {h.shape}")

    total_reward = 0.0
    for step in range(100):
        action = torch.randn(1, 3, device=device).clamp(-1, 1)
        z, h, reward, done = dream_env.step(action)
        total_reward += reward

    print(f"Dream rollout: 100 steps, total_reward={total_reward:.2f}")
    print(f"Final z: {z.shape}, range=[{z.min():.2f}, {z.max():.2f}]")
    print("Dream environment works!")
