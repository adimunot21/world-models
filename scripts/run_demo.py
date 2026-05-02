"""
Demo script — shows the full world model system in action.
Usage: python scripts/run_demo.py
"""
import argparse, sys
from pathlib import Path
import imageio, numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.transforms import resize_frame, denormalize_frame
from src.evaluation.agent import load_agent, evaluate_agent
from src.models.mdn_rnn import MDNRNN
from src.models.vae import VAE
from src.utils.logging_setup import setup_logger
logger = setup_logger("demo")

def run_and_record(agent, num_episodes=3, max_steps=500):
    import gymnasium as gym
    env = gym.make("CarRacing-v3")
    all_episodes = []
    for ep in range(num_episodes):
        obs, _ = env.reset(); agent.reset(); frames = []; total_reward = 0.0
        for step in range(max_steps):
            frames.append(obs.copy()); action = agent.act(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            if terminated or truncated: break
        all_episodes.append({"frames": frames, "reward": total_reward})
        logger.info(f"Episode {ep + 1}: reward={total_reward:.1f}, steps={len(frames)}")
    env.close(); return all_episodes

@torch.no_grad()
def generate_dream_rollout(vae, mdn_rnn, device, num_steps=100, seed=42):
    np.random.seed(seed)
    data = np.load("data/carracing/rollout_0000.npz")
    start_frame = data["observations"][50]
    frame_tensor = torch.from_numpy(start_frame.astype(np.float32) / 255.0)
    frame_tensor = frame_tensor.permute(2, 0, 1).unsqueeze(0).to(device)
    z = vae.encode_to_z(frame_tensor); hidden = mdn_rnn.init_hidden(1, device)
    dream_frames = []
    decoded = vae.decode(z).squeeze(0).permute(1, 2, 0).cpu().numpy()
    dream_frames.append(denormalize_frame(decoded))
    for step in range(num_steps):
        action = torch.randn(1, 3, device=device) * 0.3
        action[:, 0] = action[:, 0].clamp(-1, 1)
        action[:, 1] = action[:, 1].clamp(0, 1)
        action[:, 2] = action[:, 2].clamp(0, 1)
        pi, mu, log_sigma, hidden = mdn_rnn.predict_next(z, action, hidden)
        z = mdn_rnn.sample_next_z(pi, mu, log_sigma, temperature=1.0)
        decoded = vae.decode(z).squeeze(0).permute(1, 2, 0).cpu().numpy()
        dream_frames.append(denormalize_frame(decoded))
    return dream_frames

def save_video(frames, output_path, fps=30):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(output_path, fps=fps)
    for frame in frames: writer.append_data(frame)
    writer.close(); logger.info(f"Video saved to {output_path}")

def main(episodes=3, max_steps=500, save_videos=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    controller_path = Path("checkpoints/controller/best_params.npy")
    if controller_path.exists():
        logger.info("=== REAL DRIVING DEMO ===")
        params = np.load(controller_path)
        agent = load_agent(controller_params=params, device=device)
        episode_data = run_and_record(agent, num_episodes=episodes, max_steps=max_steps)
        rewards = [ep["reward"] for ep in episode_data]
        logger.info(f"Mean reward: {np.mean(rewards):.1f} +/- {np.std(rewards):.1f}")
        if save_videos and episode_data:
            save_video(episode_data[0]["frames"], "results/videos/real_driving.mp4")
    else:
        logger.warning("No controller checkpoint found.")

    vae_path, rnn_path = Path("checkpoints/vae/best.pt"), Path("checkpoints/mdn_rnn/best.pt")
    if vae_path.exists() and rnn_path.exists():
        logger.info("\n=== DREAM ROLLOUT DEMO ===")
        vae_ckpt = torch.load(vae_path, map_location=device, weights_only=False)
        vae = VAE(img_channels=3, latent_dim=vae_ckpt.get("latent_dim", 32)).to(device)
        vae.load_state_dict(vae_ckpt["model_state_dict"]); vae.eval()
        rnn_ckpt = torch.load(rnn_path, map_location=device, weights_only=False)
        cfg = rnn_ckpt.get("config", {})
        mdn_rnn = MDNRNN(latent_dim=cfg.get("latent_dim", 32), action_dim=cfg.get("action_dim", 3),
                          hidden_dim=cfg.get("hidden_dim", 256), num_gaussians=cfg.get("num_gaussians", 5)).to(device)
        mdn_rnn.load_state_dict(rnn_ckpt["model_state_dict"]); mdn_rnn.eval()
        dream_frames = generate_dream_rollout(vae, mdn_rnn, device, num_steps=100)
        logger.info(f"Generated {len(dream_frames)} dream frames")
        if save_videos: save_video(dream_frames, "results/videos/dream_rollout.mp4", fps=15)
    logger.info("\nDemo complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run world model demos.")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--no_video", action="store_true")
    args = parser.parse_args()
    main(episodes=args.episodes, max_steps=args.max_steps, save_videos=not args.no_video)
