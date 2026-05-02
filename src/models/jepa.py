"""
JEPA for CarRacing. Adapted from LeWorldModel's approach.
No decoder. Two losses: prediction MSE + SIGReg regularization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNEncoder(nn.Module):
    """CNN encoder: (B, 3, 64, 64) -> (B, embedding_dim)."""
    def __init__(self, img_channels=3, embedding_dim=192):
        super().__init__()
        self.conv1 = nn.Conv2d(img_channels, 32, 4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(128, 256, 4, stride=2, padding=1)
        self.fc = nn.Linear(256 * 4 * 4, embedding_dim)

    def forward(self, x):
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(h))
        h = h.reshape(h.size(0), -1)
        return self.fc(h)


class MLPProjector(nn.Module):
    """MLP projector with BatchNorm for comparison space."""
    def __init__(self, input_dim=192, hidden_dim=512, output_dim=192):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )
    def forward(self, x):
        return self.net(x)


class MLPPredictor(nn.Module):
    """Predicts next embedding from current embedding + action."""
    def __init__(self, embedding_dim=192, action_dim=3, hidden_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim + action_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embedding_dim),
        )
    def forward(self, emb, action):
        return self.net(torch.cat([emb, action], dim=-1))


class SIGReg(nn.Module):
    """Sketch Isotropic Gaussian Regularizer — ported from LeWorldModel."""
    def __init__(self, knots=17, num_proj=1024):
        super().__init__()
        self.num_proj = num_proj
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, embeddings):
        if embeddings.ndim == 2:
            embeddings = embeddings.unsqueeze(0)
        A = torch.randn(embeddings.size(-1), self.num_proj, device=embeddings.device)
        A = A.div_(A.norm(p=2, dim=0))
        x_t = (embeddings @ A).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights) * embeddings.size(-2)
        return statistic.mean()


class CarRacingJEPA(nn.Module):
    """Full JEPA: encoder + projectors + predictor + SIGReg."""
    def __init__(self, img_channels=3, embedding_dim=192, action_dim=3,
                 proj_hidden_dim=512, pred_hidden_dim=512,
                 sigreg_knots=17, sigreg_num_proj=1024):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.encoder = CNNEncoder(img_channels, embedding_dim)
        self.projector = MLPProjector(embedding_dim, proj_hidden_dim, embedding_dim)
        self.predictor = MLPPredictor(embedding_dim, action_dim, pred_hidden_dim)
        self.pred_proj = MLPProjector(embedding_dim, proj_hidden_dim, embedding_dim)
        self.sigreg = SIGReg(sigreg_knots, sigreg_num_proj)

    def encode(self, frames):
        if frames.ndim == 5:
            B, T = frames.shape[:2]
            flat = frames.reshape(B * T, *frames.shape[2:])
            emb = self.encoder(flat)
            return emb.reshape(B, T, -1)
        return self.encoder(frames)

    def forward(self, frames, actions):
        B, Tp1 = frames.shape[:2]
        T = Tp1 - 1

        raw_emb = self.encode(frames)
        flat_emb = raw_emb.reshape(B * Tp1, -1)
        proj_emb = self.projector(flat_emb).reshape(B, Tp1, -1)

        emb_input = raw_emb[:, :T, :]
        flat_input = emb_input.reshape(B * T, -1)
        flat_actions = actions.reshape(B * T, -1)

        flat_pred = self.predictor(flat_input, flat_actions)
        flat_pred_proj = self.pred_proj(flat_pred)
        pred_emb = flat_pred_proj.reshape(B, T, -1)

        return {
            "projected_emb": proj_emb,
            "predicted_emb": pred_emb,
            "raw_emb": raw_emb,
        }


def jepa_loss(output, sigreg, sigreg_weight=1.0):
    proj_emb = output["projected_emb"]
    pred_emb = output["predicted_emb"]
    raw_emb = output["raw_emb"]

    target_emb = proj_emb[:, 1:, :].detach()
    pred_loss = F.mse_loss(pred_emb, target_emb)

    flat_raw = raw_emb.reshape(1, -1, raw_emb.size(-1))
    sig_loss = sigreg(flat_raw)

    total_loss = pred_loss + sigreg_weight * sig_loss
    return total_loss, pred_loss, sig_loss


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CarRacingJEPA(embedding_dim=192, action_dim=3).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    enc_params = sum(p.numel() for p in model.encoder.parameters())
    pred_params = sum(p.numel() for p in model.predictor.parameters())
    print(f"Total params:     {total_params:,}")
    print(f"Encoder params:   {enc_params:,}")
    print(f"Predictor params: {pred_params:,}")

    B, T = 4, 8
    frames = torch.randn(B, T + 1, 3, 64, 64, device=device).clamp(0, 1)
    actions = torch.randn(B, T, 3, device=device)

    output = model(frames, actions)
    print(f"\nForward pass:")
    for k, v in output.items():
        print(f"  {k}: {v.shape}")

    loss, pred_loss, sig_loss = jepa_loss(output, model.sigreg, sigreg_weight=1.0)
    print(f"\nLoss: {loss.item():.4f} (pred={pred_loss.item():.4f}, sig={sig_loss.item():.4f})")

    mem_mb = torch.cuda.max_memory_allocated() / 1e6
    print(f"Peak VRAM: {mem_mb:.0f} MB")
