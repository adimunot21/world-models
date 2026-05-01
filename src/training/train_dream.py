"""
Dream training — train controller entirely inside MDN-RNN imagination.
Periodically evaluates in real CarRacing to measure transfer quality.

Usage:
  python src/training/train_dream.py
  python src/training/train_dream.py --temperature 1.15 --max_generations 300
"""

import argparse
import sys
import time
from pathlib import Path

import cma
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.evaluation.agent import load_agent, evaluate_agent
from src.evaluation.dream_env import collect_initial_z, evaluate_in_dream
from src.models.controller import Controller
from src.models.mdn_rnn import MDNRNN
from src.utils.logging_setup import setup_logger
from src.utils.reproducibility import set_seed

logger = setup_logger("train_dream")


def train_dream(
    mdn_rnn_checkpoint="checkpoints/mdn_rnn/best.pt",
    vae_checkpoint="checkpoints/vae/best.pt",
    encoded_dir="data/carracing/encoded",
    save_dir="checkpoints/controller",
    pop_size=32,
    sigma_init=0.1,
    max_generations=300,
    num_rollouts=5,
    max_dream_steps=300,
    temperature=1.0,
    seed=42,
    real_eval_every=25,
    real_eval_episodes=5,
    real_eval_max_steps=1000,
):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Load MDN-RNN
    rnn_ckpt = torch.load(mdn_rnn_checkpoint, map_location=device, weights_only=False)
    rnn_config = rnn_ckpt.get("config", {})
    latent_dim = rnn_config.get("latent_dim", 32)
    hidden_dim = rnn_config.get("hidden_dim", 256)

    mdn_rnn = MDNRNN(
        latent_dim=latent_dim,
        action_dim=rnn_config.get("action_dim", 3),
        hidden_dim=hidden_dim,
        num_gaussians=rnn_config.get("num_gaussians", 5),
    ).to(device)
    mdn_rnn.load_state_dict(rnn_ckpt["model_state_dict"])
    mdn_rnn.eval()
    logger.info(f"Loaded MDN-RNN from {mdn_rnn_checkpoint}")

    # Collect initial z pool
    z_pool = collect_initial_z(encoded_dir, num_rollouts=100, samples_per_rollout=10)

    # Create controller
    controller = Controller(latent_dim=latent_dim, hidden_dim=hidden_dim, action_dim=3).to(device)
    num_params = controller.get_num_params()
    logger.info(f"Controller parameters: {num_params}")
    logger.info(
        f"Dream config: temp={temperature}, dream_steps={max_dream_steps}, "
        f"dream_rollouts={num_rollouts}"
    )

    # CMA-ES setup
    x0 = np.zeros(num_params)
    es = cma.CMAEvolutionStrategy(
        x0, sigma_init,
        {"popsize": pop_size, "seed": seed, "maxiter": max_generations},
    )

    temp_str = f"temp_{temperature:.2f}".replace(".", "p")
    save_path = Path(save_dir) / f"dream_{temp_str}"
    save_path.mkdir(parents=True, exist_ok=True)

    best_dream_reward = -float("inf")
    best_params_ever = None
    real_eval_results = []

    logger.info("Starting dream training...")
    gen = 0
    start_time = time.time()

    while not es.stop():
        gen += 1
        gen_start = time.time()

        candidates = es.ask()
        fitnesses = []
        rewards_this_gen = []

        for params in candidates:
            reward = evaluate_in_dream(
                controller, mdn_rnn, z_pool, device,
                params, num_rollouts, max_dream_steps, temperature,
            )
            fitnesses.append(-reward)
            rewards_this_gen.append(reward)

        es.tell(candidates, fitnesses)

        gen_mean = np.mean(rewards_this_gen)
        gen_best = np.max(rewards_this_gen)
        gen_time = time.time() - gen_start

        gen_best_idx = np.argmax(rewards_this_gen)
        if gen_best > best_dream_reward:
            best_dream_reward = gen_best
            best_params_ever = candidates[gen_best_idx].copy()
            np.save(save_path / "best_params.npy", best_params_ever)

        elapsed = time.time() - start_time
        logger.info(
            f"Gen {gen:4d} | "
            f"dream_mean={gen_mean:7.1f} | "
            f"dream_best={gen_best:7.1f} | "
            f"best_ever={best_dream_reward:7.1f} | "
            f"sigma={es.sigma:.4f} | "
            f"{gen_time:.1f}s | "
            f"elapsed={elapsed / 60:.0f}min"
        )

        if gen % 10 == 0:
            np.save(save_path / f"gen_{gen:04d}_params.npy", best_params_ever)

        # Periodic real-environment evaluation
        if gen % real_eval_every == 0 and best_params_ever is not None:
            logger.info(f"  Real-env evaluation ({real_eval_episodes} episodes)...")
            agent = load_agent(
                vae_checkpoint=vae_checkpoint,
                mdn_rnn_checkpoint=mdn_rnn_checkpoint,
                controller_params=best_params_ever,
                device=device,
            )
            results = evaluate_agent(
                agent, num_episodes=real_eval_episodes, max_steps=real_eval_max_steps,
            )
            real_eval_results.append({
                "generation": gen,
                "mean_reward": results["mean_reward"],
                "std_reward": results["std_reward"],
                "rewards": results["rewards"],
            })
            logger.info(
                f"  Real-env: mean={results['mean_reward']:.1f} "
                f"+/- {results['std_reward']:.1f} | "
                f"rewards={[f'{r:.0f}' for r in results['rewards']]}"
            )

    # Final save
    torch.save(
        {
            "params": best_params_ever,
            "dream_reward": best_dream_reward,
            "generation": gen,
            "temperature": temperature,
            "real_eval_results": real_eval_results,
            "config": {
                "pop_size": pop_size, "sigma_init": sigma_init,
                "num_rollouts": num_rollouts, "max_dream_steps": max_dream_steps,
                "temperature": temperature,
            },
        },
        save_path / "final.pt",
    )
    np.save(save_path / "final_params.npy", best_params_ever)

    total_time = time.time() - start_time
    logger.info(
        f"Dream training complete. {gen} generations, "
        f"best_dream_reward={best_dream_reward:.1f}, "
        f"total_time={total_time / 60:.0f} min"
    )

    # Final comprehensive real-env evaluation
    logger.info("Final real-env evaluation: 10 episodes...")
    agent = load_agent(
        vae_checkpoint=vae_checkpoint,
        mdn_rnn_checkpoint=mdn_rnn_checkpoint,
        controller_params=best_params_ever,
        device=device,
    )
    final_results = evaluate_agent(agent, num_episodes=10, max_steps=1000)
    for i, r in enumerate(final_results["rewards"]):
        logger.info(f"  Episode {i + 1}: reward={r:.1f}")
    logger.info(
        f"Final real-env: mean={final_results['mean_reward']:.1f} "
        f"+/- {final_results['std_reward']:.1f}"
    )

    torch.save(
        {
            "dream_reward": best_dream_reward,
            "real_rewards": final_results["rewards"],
            "real_mean": final_results["mean_reward"],
            "real_std": final_results["std_reward"],
            "temperature": temperature,
            "eval_history": real_eval_results,
        },
        save_path / "transfer_results.pt",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train controller in MDN-RNN dreams.")
    parser.add_argument("--mdn_rnn_checkpoint", type=str, default="checkpoints/mdn_rnn/best.pt")
    parser.add_argument("--vae_checkpoint", type=str, default="checkpoints/vae/best.pt")
    parser.add_argument("--encoded_dir", type=str, default="data/carracing/encoded")
    parser.add_argument("--save_dir", type=str, default="checkpoints/controller")
    parser.add_argument("--pop_size", type=int, default=32)
    parser.add_argument("--sigma_init", type=float, default=0.1)
    parser.add_argument("--max_generations", type=int, default=300)
    parser.add_argument("--num_rollouts", type=int, default=5)
    parser.add_argument("--max_dream_steps", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--real_eval_every", type=int, default=25)
    parser.add_argument("--real_eval_episodes", type=int, default=5)
    args = parser.parse_args()

    train_dream(
        mdn_rnn_checkpoint=args.mdn_rnn_checkpoint,
        vae_checkpoint=args.vae_checkpoint,
        encoded_dir=args.encoded_dir,
        save_dir=args.save_dir,
        pop_size=args.pop_size,
        sigma_init=args.sigma_init,
        max_generations=args.max_generations,
        num_rollouts=args.num_rollouts,
        max_dream_steps=args.max_dream_steps,
        temperature=args.temperature,
        seed=args.seed,
        real_eval_every=args.real_eval_every,
        real_eval_episodes=args.real_eval_episodes,
    )
