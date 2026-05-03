"""MDN-RNN training with reward prediction."""
import argparse, sys, time
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
    loss_sum = 0.0; num_batches = 0
    for batch in dataloader:
        z_seq = batch["z"].to(device)
        actions = batch["actions"].to(device)
        rewards = batch["rewards"].to(device)

        z_input = z_seq[:, :-1, :]
        z_target = z_seq[:, 1:, :]

        pi, mu, log_sigma, reward_pred, _ = model(z_input, actions)
        loss = mdn_loss(z_target, pi, mu, log_sigma, reward_pred, rewards)

        if torch.isnan(loss):
            logger.warning("NaN loss, skipping batch"); continue
        optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        loss_sum += loss.item(); num_batches += 1
    if num_batches == 0: return {"loss": float("nan")}
    return {"loss": loss_sum / num_batches}

@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    loss_sum = 0.0; num_batches = 0
    for batch in dataloader:
        z_seq = batch["z"].to(device)
        actions = batch["actions"].to(device)
        rewards = batch["rewards"].to(device)

        z_input = z_seq[:, :-1, :]
        z_target = z_seq[:, 1:, :]

        pi, mu, log_sigma, reward_pred, _ = model(z_input, actions)
        loss = mdn_loss(z_target, pi, mu, log_sigma, reward_pred, rewards)

        if not torch.isnan(loss):
            loss_sum += loss.item(); num_batches += 1
    if num_batches == 0: return {"loss": float("nan")}
    return {"loss": loss_sum / num_batches}

def train(data_dir="data/carracing/encoded", epochs=50, batch_size=64, seq_len=32,
          learning_rate=1e-3, latent_dim=32, action_dim=3, hidden_dim=256,
          num_gaussians=5, grad_clip=1.0, save_dir="checkpoints/mdn_rnn",
          save_every=10, seed=42, num_workers=4, val_split=0.1):
    set_seed(seed); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    dataset = LatentSequenceDataset(data_dir, seq_len=seq_len)
    val_size = int(len(dataset) * val_split); train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    logger.info(f"Train: {train_size:,}, Val: {val_size:,}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)

    model = MDNRNN(latent_dim=latent_dim, action_dim=action_dim,
                   hidden_dim=hidden_dim, num_gaussians=num_gaussians).to(device)
    logger.info(f"MDN-RNN parameters: {sum(p.numel() for p in model.parameters()):,}")
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    save_path = Path(save_dir); save_path.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir="runs/mdn_rnn_v2")
    best_val_loss = float("inf"); nan_streak = 0

    logger.info(f"Starting training: {epochs} epochs (now with reward prediction)")

    for epoch in range(epochs):
        t0 = time.time()
        train_m = train_one_epoch(model, train_loader, optimizer, device, grad_clip)
        val_m = evaluate(model, val_loader, device)
        dt = time.time() - t0

        if torch.isnan(torch.tensor(train_m["loss"])):
            nan_streak += 1
            if nan_streak >= 5: logger.error("Too many NaN. Stopping."); break
            continue
        nan_streak = 0

        writer.add_scalar("train/loss", train_m["loss"], epoch)
        writer.add_scalar("val/loss", val_m["loss"], epoch)
        logger.info(f"Epoch {epoch+1:3d}/{epochs} | train={train_m['loss']:8.2f} | val={val_m['loss']:8.2f} | {dt:.1f}s")

        if val_m["loss"] < best_val_loss:
            best_val_loss = val_m["loss"]
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_loss": best_val_loss,
                        "config": {"latent_dim": latent_dim, "action_dim": action_dim,
                                   "hidden_dim": hidden_dim, "num_gaussians": num_gaussians,
                                   "seq_len": seq_len}}, save_path / "best.pt")
            logger.info(f"  -> New best (val={best_val_loss:.2f})")

        if (epoch + 1) % save_every == 0:
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict()},
                       save_path / f"epoch_{epoch+1:03d}.pt")

    torch.save({"epoch": epochs-1, "model_state_dict": model.state_dict(),
                "config": {"latent_dim": latent_dim, "action_dim": action_dim,
                           "hidden_dim": hidden_dim, "num_gaussians": num_gaussians}},
               save_path / "final.pt")
    writer.close()
    logger.info(f"Training complete. Best val={best_val_loss:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/carracing/encoded")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seq_len", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--latent_dim", type=int, default=32)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_gaussians", type=int, default=5)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--save_dir", default="checkpoints/mdn_rnn")
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()
    train(**vars(args))
