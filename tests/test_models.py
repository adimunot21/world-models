"""Tests for core model components. Run: pytest tests/test_models.py -v"""
import sys
from pathlib import Path
import numpy as np, pytest, torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.models.vae import VAE, vae_loss
from src.models.mdn_rnn import MDNRNN, mdn_loss
from src.models.controller import Controller
from src.models.jepa import CarRacingJEPA, SIGReg, jepa_loss

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TestVAE:
    def setup_method(self):
        self.model = VAE(img_channels=3, latent_dim=32).to(DEVICE)
        self.x = torch.randn(2, 3, 64, 64, device=DEVICE).clamp(0, 1)
    def test_forward_shapes(self):
        x_recon, mu, log_var, z = self.model(self.x)
        assert x_recon.shape == (2, 3, 64, 64)
        assert mu.shape == (2, 32)
        assert z.shape == (2, 32)
    def test_reconstruction_range(self):
        x_recon, _, _, _ = self.model(self.x)
        assert x_recon.min() >= 0.0 and x_recon.max() <= 1.0
    def test_encode_decode_shapes(self):
        z = self.model.encode_to_z(self.x)
        assert z.shape == (2, 32)
        decoded = self.model.decode(z)
        assert decoded.shape == (2, 3, 64, 64)
    def test_loss_not_nan(self):
        x_recon, mu, log_var, z = self.model(self.x)
        loss, recon, kl = vae_loss(self.x, x_recon, mu, log_var)
        assert not torch.isnan(loss)
    def test_kl_zero_at_prior(self):
        mu = torch.zeros(2, 32, device=DEVICE)
        log_var = torch.zeros(2, 32, device=DEVICE)
        _, _, kl = vae_loss(self.x, self.x, mu, log_var)
        assert kl.item() < 0.01

class TestMDNRNN:
    def setup_method(self):
        self.model = MDNRNN(latent_dim=32, action_dim=3, hidden_dim=64, num_gaussians=3).to(DEVICE)
    def test_forward_shapes(self):
        z = torch.randn(2, 8, 32, device=DEVICE)
        a = torch.randn(2, 8, 3, device=DEVICE)
        pi, mu, log_sigma, hidden = self.model(z, a)
        assert pi.shape == (2, 8, 3)
        assert mu.shape == (2, 8, 3, 32)
    def test_pi_sums_to_one(self):
        z = torch.randn(2, 8, 32, device=DEVICE)
        a = torch.randn(2, 8, 3, device=DEVICE)
        pi, _, _, _ = self.model(z, a)
        assert torch.allclose(pi.sum(dim=-1), torch.ones(2, 8, device=DEVICE), atol=1e-5)
    def test_sampling(self):
        z_t = torch.randn(2, 32, device=DEVICE)
        a_t = torch.randn(2, 3, device=DEVICE)
        hidden = self.model.init_hidden(2, DEVICE)
        pi, mu, log_sigma, _ = self.model.predict_next(z_t, a_t, hidden)
        z_next = self.model.sample_next_z(pi, mu, log_sigma)
        assert z_next.shape == (2, 32)
    def test_loss_not_nan(self):
        z = torch.randn(2, 8, 32, device=DEVICE)
        a = torch.randn(2, 8, 3, device=DEVICE)
        pi, mu, log_sigma, _ = self.model(z, a)
        loss = mdn_loss(torch.randn(2, 8, 32, device=DEVICE), pi, mu, log_sigma)
        assert not torch.isnan(loss)

class TestController:
    def setup_method(self):
        self.ctrl = Controller(latent_dim=32, hidden_dim=64, action_dim=3)
    def test_param_count(self):
        assert self.ctrl.get_num_params() == (32 + 64) * 3 + 3
    def test_output_ranges(self):
        action = self.ctrl(torch.randn(4, 32), torch.randn(4, 64))
        assert action[:, 0].min() >= -1.0 and action[:, 0].max() <= 1.0
        assert action[:, 1].min() >= 0.0 and action[:, 2].max() <= 1.0
    def test_flat_param_roundtrip(self):
        new = np.random.randn(self.ctrl.get_num_params()).astype(np.float32)
        self.ctrl.set_flat_params(new)
        assert np.allclose(new, self.ctrl.get_flat_params(), atol=1e-6)

class TestJEPA:
    def setup_method(self):
        self.model = CarRacingJEPA(embedding_dim=64, action_dim=3,
                                    proj_hidden_dim=128, pred_hidden_dim=128).to(DEVICE)
    def test_encode_single(self):
        emb = self.model.encode(torch.randn(2, 3, 64, 64, device=DEVICE))
        assert emb.shape == (2, 64)
    def test_encode_sequence(self):
        emb = self.model.encode(torch.randn(2, 4, 3, 64, 64, device=DEVICE))
        assert emb.shape == (2, 4, 64)
    def test_forward_shapes(self):
        output = self.model(torch.randn(2, 5, 3, 64, 64, device=DEVICE),
                           torch.randn(2, 4, 3, device=DEVICE))
        assert output["projected_emb"].shape == (2, 5, 64)
        assert output["predicted_emb"].shape == (2, 4, 64)
    def test_loss_not_nan(self):
        output = self.model(torch.randn(2, 5, 3, 64, 64, device=DEVICE).clamp(0,1),
                           torch.randn(2, 4, 3, device=DEVICE))
        loss, _, _ = jepa_loss(output, self.model.sigreg)
        assert not torch.isnan(loss)

class TestSIGReg:
    def test_gaussian_low_loss(self):
        sigreg = SIGReg(knots=17, num_proj=512).to(DEVICE)
        loss = sigreg(torch.randn(1, 256, 64, device=DEVICE))
        assert loss.item() < 5.0
    def test_constant_high_loss(self):
        sigreg = SIGReg(knots=17, num_proj=512).to(DEVICE)
        loss = sigreg(torch.ones(1, 256, 64, device=DEVICE))
        assert loss.item() > 10.0

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
