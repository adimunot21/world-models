"""
MDN-RNN with reward prediction head.

Now predicts BOTH next latent state (mixture of Gaussians) AND reward.
The reward head enables realistic dream training — the dream environment
gives accurate reward signals instead of heuristics.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

LOG_SIGMA_MIN = -7.0
LOG_SIGMA_MAX = 2.0


class MDNRNN(nn.Module):
    def __init__(self, latent_dim=32, action_dim=3, hidden_dim=256,
                 num_gaussians=5, num_layers=1):
        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.num_gaussians = num_gaussians
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=latent_dim + action_dim, hidden_size=hidden_dim,
            num_layers=num_layers, batch_first=True,
        )

        output_size = num_gaussians + 2 * num_gaussians * latent_dim
        self.mdn_head = nn.Linear(hidden_dim, output_size)

        # Reward prediction head
        self.reward_head = nn.Linear(hidden_dim, 1)

    def forward(self, z, actions, hidden=None):
        B, T, _ = z.shape
        K = self.num_gaussians
        D = self.latent_dim

        lstm_input = torch.cat([z, actions], dim=-1)
        if hidden is None:
            lstm_out, hidden = self.lstm(lstm_input)
        else:
            lstm_out, hidden = self.lstm(lstm_input, hidden)

        mdn_out = self.mdn_head(lstm_out)
        pi_logits = mdn_out[:, :, :K]
        remaining = mdn_out[:, :, K:].reshape(B, T, 2 * K, D)
        mu = remaining[:, :, :K, :]
        log_sigma = remaining[:, :, K:, :].clamp(LOG_SIGMA_MIN, LOG_SIGMA_MAX)
        pi = F.softmax(pi_logits, dim=-1)

        reward_pred = self.reward_head(lstm_out).squeeze(-1)  # (B, T)

        return pi, mu, log_sigma, reward_pred, hidden

    def init_hidden(self, batch_size, device):
        h = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
        c = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
        return (h, c)

    def predict_next(self, z_t, action_t, hidden):
        z_t = z_t.unsqueeze(1)
        action_t = action_t.unsqueeze(1)
        pi, mu, log_sigma, reward_pred, hidden = self.forward(z_t, action_t, hidden)
        return pi.squeeze(1), mu.squeeze(1), log_sigma.squeeze(1), reward_pred.squeeze(1), hidden

    def sample_next_z(self, pi, mu, log_sigma, temperature=1.0):
        B = pi.shape[0]
        if temperature != 1.0:
            pi = F.softmax(torch.log(pi + 1e-8) / temperature, dim=-1)
        component_idx = torch.multinomial(pi, 1).squeeze(-1)
        idx = component_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, self.latent_dim)
        selected_mu = mu.gather(1, idx).squeeze(1)
        selected_log_sigma = log_sigma.gather(1, idx).squeeze(1)
        sigma = torch.exp(selected_log_sigma) * temperature
        eps = torch.randn_like(selected_mu)
        return selected_mu + sigma * eps


def mdn_loss(z_target, pi, mu, log_sigma, reward_pred=None, reward_target=None, reward_weight=1.0):
    z_expanded = z_target.unsqueeze(2)
    sigma = torch.exp(log_sigma)
    log_normal = (
        -0.5 * torch.log(torch.tensor(2.0 * torch.pi, device=z_target.device))
        - log_sigma
        - 0.5 * ((z_expanded - mu) / (sigma + 1e-8)) ** 2
    )
    log_prob_per_component = log_normal.sum(dim=-1)
    log_pi = torch.log(pi + 1e-8)
    log_weighted = log_pi + log_prob_per_component
    log_prob = torch.logsumexp(log_weighted, dim=-1)
    nll = -log_prob.mean()

    if reward_pred is not None and reward_target is not None:
        reward_loss = F.mse_loss(reward_pred, reward_target)
        return nll + reward_weight * reward_loss
    return nll


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MDNRNN(latent_dim=32, action_dim=3, hidden_dim=256, num_gaussians=5).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total params: {total_params:,}")

    B, T, D, A = 4, 32, 32, 3
    z = torch.randn(B, T, D, device=device)
    actions = torch.randn(B, T, A, device=device)
    pi, mu, log_sigma, reward_pred, hidden = model(z, actions)
    print(f"\nSequence forward pass:")
    print(f"  pi:          {pi.shape} (sum={pi[0, 0].sum().item():.4f})")
    print(f"  mu:          {mu.shape}")
    print(f"  reward_pred: {reward_pred.shape}")

    z_target = torch.randn(B, T, D, device=device)
    r_target = torch.randn(B, T, device=device)
    loss = mdn_loss(z_target, pi, mu, log_sigma, reward_pred, r_target)
    print(f"  loss (with reward): {loss.item():.2f}")

    z_t = torch.randn(B, D, device=device)
    a_t = torch.randn(B, A, device=device)
    hidden = model.init_hidden(B, device)
    pi, mu, log_sigma, r_pred, hidden = model.predict_next(z_t, a_t, hidden)
    print(f"\nSingle-step: pi={pi.shape}, reward={r_pred.shape}")
    z_next = model.sample_next_z(pi, mu, log_sigma)
    print(f"  z_next: {z_next.shape}")
    print(f"Peak VRAM: {torch.cuda.max_memory_allocated() / 1e6:.0f} MB")
