"""
World Model Agent — integrates VAE + MDN-RNN + Controller.

Pipeline per step:
  raw frame -> VAE encode -> z -> MDN-RNN update hidden -> Controller([z, h]) -> action
"""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.transforms import resize_frame
from src.models.controller import Controller
from src.models.mdn_rnn import MDNRNN
from src.models.vae import VAE


class WorldModelAgent:
    """Combines VAE + MDN-RNN + Controller into a full agent."""

    def __init__(self, vae, mdn_rnn, controller, device, img_size=64):
        self.vae = vae.to(device).eval()
        self.mdn_rnn = mdn_rnn.to(device).eval()
        self.controller = controller.to(device).eval()
        self.device = device
        self.img_size = img_size
        self.hidden = None
        self.prev_z = None
        self.prev_action = None

    def reset(self):
        """Reset agent state for a new episode."""
        self.hidden = self.mdn_rnn.init_hidden(1, self.device)
        self.prev_z = None
        self.prev_action = None

    @torch.no_grad()
    def act(self, observation: np.ndarray) -> np.ndarray:
        """raw frame (96x96x3 uint8) -> action (3,) float32."""
        frame = resize_frame(observation, self.img_size)
        frame = frame.astype(np.float32) / 255.0
        frame = np.transpose(frame, (2, 0, 1))
        frame_tensor = torch.from_numpy(frame).unsqueeze(0).to(self.device)

        z = self.vae.encode_to_z(frame_tensor)

        if self.prev_z is not None and self.prev_action is not None:
            _, _, _, self.hidden = self.mdn_rnn.predict_next(
                self.prev_z, self.prev_action, self.hidden
            )

        h = self.hidden[0][-1]
        action = self.controller(z, h)
        action_np = action.squeeze(0).cpu().numpy()

        self.prev_z = z
        self.prev_action = action

        return action_np


def load_agent(
    vae_checkpoint="checkpoints/vae/best.pt",
    mdn_rnn_checkpoint="checkpoints/mdn_rnn/best.pt",
    controller_params=None,
    device=None,
):
    """Load a complete agent from checkpoints."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vae_ckpt = torch.load(vae_checkpoint, map_location=device, weights_only=False)
    latent_dim = vae_ckpt.get("latent_dim", 32)
    vae = VAE(img_channels=3, latent_dim=latent_dim)
    vae.load_state_dict(vae_ckpt["model_state_dict"])

    rnn_ckpt = torch.load(mdn_rnn_checkpoint, map_location=device, weights_only=False)
    rnn_config = rnn_ckpt.get("config", {})
    mdn_rnn = MDNRNN(
        latent_dim=rnn_config.get("latent_dim", 32),
        action_dim=rnn_config.get("action_dim", 3),
        hidden_dim=rnn_config.get("hidden_dim", 256),
        num_gaussians=rnn_config.get("num_gaussians", 5),
    )
    mdn_rnn.load_state_dict(rnn_ckpt["model_state_dict"])

    controller = Controller(
        latent_dim=latent_dim,
        hidden_dim=rnn_config.get("hidden_dim", 256),
        action_dim=3,
    )
    if controller_params is not None:
        controller.set_flat_params(controller_params)

    return WorldModelAgent(vae, mdn_rnn, controller, device)


def evaluate_agent(agent, num_episodes=5, max_steps=1000, render=False):
    """Run agent in CarRacing, return performance metrics."""
    import gymnasium as gym

    env = gym.make("CarRacing-v3", render_mode="human" if render else None)
    rewards = []

    for ep in range(num_episodes):
        obs, _ = env.reset()
        agent.reset()
        total_reward = 0.0

        for step in range(max_steps):
            action = agent.act(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            if terminated or truncated:
                break

        rewards.append(total_reward)

    env.close()
    return {
        "rewards": rewards,
        "mean_reward": np.mean(rewards),
        "std_reward": np.std(rewards),
        "num_episodes": num_episodes,
    }


if __name__ == "__main__":
    print("Loading agent with random controller...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        agent = load_agent(device=device)
        print(f"Agent loaded. Controller params: {agent.controller.get_num_params()}")

        print("Running 1 episode with random controller...")
        results = evaluate_agent(agent, num_episodes=1, max_steps=200)
        print(f"Reward: {results['rewards'][0]:.1f} (random controller, expected to be bad)")
    except FileNotFoundError as e:
        print(f"Checkpoint not found: {e}")
        print("Run VAE and MDN-RNN training first.")
