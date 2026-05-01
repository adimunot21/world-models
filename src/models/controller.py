"""
Linear controller for the world model agent.

Input:  [z (32-dim), h (256-dim)] = 288-dim
Output: 3 actions [steering, gas, brake]
Total:  288 * 3 + 3 = 867 parameters

Intentionally tiny — all intelligence comes from the VAE + MDN-RNN
representations. Trained by CMA-ES (evolutionary, no gradients).
"""

import numpy as np
import torch
import torch.nn as nn


class Controller(nn.Module):
    def __init__(self, latent_dim: int = 32, hidden_dim: int = 256, action_dim: int = 3):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.fc = nn.Linear(latent_dim + hidden_dim, action_dim)

    def forward(self, z: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """[z, h] -> [steering (tanh), gas (sigmoid), brake (sigmoid)]."""
        x = torch.cat([z, h], dim=-1)
        raw = self.fc(x)
        steering = torch.tanh(raw[:, 0:1])
        gas = torch.sigmoid(raw[:, 1:2])
        brake = torch.sigmoid(raw[:, 2:3])
        return torch.cat([steering, gas, brake], dim=-1)

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def get_flat_params(self) -> np.ndarray:
        """Flatten all params to 1D numpy array for CMA-ES."""
        params = []
        for p in self.parameters():
            params.append(p.data.cpu().numpy().flatten())
        return np.concatenate(params)

    def set_flat_params(self, flat_params: np.ndarray) -> None:
        """Load params from 1D numpy array (inverse of get_flat_params)."""
        idx = 0
        for p in self.parameters():
            n = p.numel()
            p.data.copy_(
                torch.from_numpy(flat_params[idx : idx + n].reshape(p.shape)).float()
            )
            idx += n


if __name__ == "__main__":
    ctrl = Controller(latent_dim=32, hidden_dim=256, action_dim=3)
    print(f"Parameters: {ctrl.get_num_params()}")

    z = torch.randn(1, 32)
    h = torch.randn(1, 256)
    action = ctrl(z, h)
    print(f"Action: {action.detach().numpy()}")
    print(f"  steering: {action[0, 0].item():.3f} (should be in [-1, 1])")
    print(f"  gas:      {action[0, 1].item():.3f} (should be in [0, 1])")
    print(f"  brake:    {action[0, 2].item():.3f} (should be in [0, 1])")

    flat = ctrl.get_flat_params()
    print(f"\nFlat params: shape={flat.shape}, range=[{flat.min():.3f}, {flat.max():.3f}]")

    new_params = np.random.randn(flat.shape[0]) * 0.1
    ctrl.set_flat_params(new_params)
    recovered = ctrl.get_flat_params()
    assert np.allclose(new_params, recovered), "Round-trip failed!"
    print("Flat param round-trip: OK")
