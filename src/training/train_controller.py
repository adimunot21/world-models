"""
CMA-ES training for the controller.

CMA-ES maintains a Gaussian distribution over the 867-dim parameter space
and iteratively evolves toward high-reward solutions. No gradients.

Usage:
  python src/training/train_controller.py
  python src/training/train_controller.py --pop_size 32 --max_generations 200
"""

import argparse
import sys
import time
from pathlib import Path

import cma
import gymnasium as gym
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.evaluation.agent import WorldModelAgent, load_agent
from src.models.controller import Controller
from src.utils.logging_setup import setup_logger
from src.utils.reproducibility import set_seed

logger = setup_logger("train_ctrl")


def evaluate_single(agent, controller_params, num_rollouts=3, max_steps=1000):
    """Evaluate a single controller: set weights, run episodes, return mean reward."""
    agent.controller.set_flat_params(controller_params)
    env = gym.make("CarRacing-v3")
    rewards = []

    for _ in range(num_rollouts):
        obs, _ = env.reset()
        agent.reset()
        total_reward = 0.0

        for _ in range(max_steps):
            action = agent.act(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            if terminated or truncated:
                break

        rewards.append(total_reward)

    env.close()
    return float(np.mean(rewards))


def train_controller(
    vae_checkpoint="checkpoints/vae/best.pt",
    mdn_rnn_checkpoint="checkpoints/mdn_rnn/best.pt",
    save_dir="checkpoints/controller",
    pop_size=32,
    sigma_init=0.1,
    max_generations=200,
    target_reward=800.0,
    num_rollouts=3,
    max_steps=1000,
    seed=42,
):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    agent = load_agent(
        vae_checkpoint=vae_checkpoint,
        mdn_rnn_checkpoint=mdn_rnn_checkpoint,
        device=device,
    )
    num_params = agent.controller.get_num_params()
    logger.info(f"Controller parameters: {num_params}")
    logger.info(
        f"CMA-ES config: pop_size={pop_size}, sigma={sigma_init}, "
        f"max_gen={max_generations}, target={target_reward}"
    )
    logger.info(f"Evaluation: {num_rollouts} rollouts x {max_steps} steps per candidate")

    x0 = np.zeros(num_params)
    es = cma.CMAEvolutionStrategy(
        x0, sigma_init,
        {"popsize": pop_size, "seed": seed, "maxiter": max_generations},
    )

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    best_reward_ever = -float("inf")
    best_params_ever = None

    logger.info("Starting CMA-ES optimization...")
    gen = 0
    start_time = time.time()

    while not es.stop():
        gen += 1
        gen_start = time.time()

        candidates = es.ask()

        fitnesses = []
        rewards_this_gen = []

        for i, params in enumerate(candidates):
            reward = evaluate_single(agent, params, num_rollouts, max_steps)
            fitnesses.append(-reward)
            rewards_this_gen.append(reward)

        es.tell(candidates, fitnesses)

        gen_mean_reward = np.mean(rewards_this_gen)
        gen_best_reward = np.max(rewards_this_gen)
        gen_worst_reward = np.min(rewards_this_gen)
        gen_time = time.time() - gen_start

        gen_best_idx = np.argmax(rewards_this_gen)
        if gen_best_reward > best_reward_ever:
            best_reward_ever = gen_best_reward
            best_params_ever = candidates[gen_best_idx].copy()

            np.save(save_path / "best_params.npy", best_params_ever)
            torch.save(
                {
                    "params": best_params_ever,
                    "reward": best_reward_ever,
                    "generation": gen,
                    "config": {
                        "pop_size": pop_size, "sigma_init": sigma_init,
                        "num_rollouts": num_rollouts, "max_steps": max_steps,
                    },
                },
                save_path / "best.pt",
            )

        elapsed = time.time() - start_time
        logger.info(
            f"Gen {gen:4d} | "
            f"mean={gen_mean_reward:7.1f} | "
            f"best={gen_best_reward:7.1f} | "
            f"worst={gen_worst_reward:7.1f} | "
            f"best_ever={best_reward_ever:7.1f} | "
            f"sigma={es.sigma:.4f} | "
            f"{gen_time:.0f}s | "
            f"elapsed={elapsed / 60:.0f}min"
        )

        if gen % 10 == 0:
            np.save(save_path / f"gen_{gen:04d}_params.npy", best_params_ever)

        if best_reward_ever >= target_reward:
            logger.info(f"Target reward {target_reward} reached! Stopping.")
            break

    if best_params_ever is not None:
        np.save(save_path / "final_params.npy", best_params_ever)
        torch.save(
            {"params": best_params_ever, "reward": best_reward_ever, "generation": gen},
            save_path / "final.pt",
        )

    total_time = time.time() - start_time
    logger.info(
        f"CMA-ES complete. {gen} generations, "
        f"best_reward={best_reward_ever:.1f}, total_time={total_time / 60:.0f} min"
    )

    # Final evaluation
    logger.info("Final evaluation: 10 episodes with best controller...")
    agent.controller.set_flat_params(best_params_ever)
    env = gym.make("CarRacing-v3")
    final_rewards = []
    for ep in range(10):
        obs, _ = env.reset()
        agent.reset()
        total_reward = 0.0
        for _ in range(max_steps):
            action = agent.act(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        final_rewards.append(total_reward)
        logger.info(f"  Episode {ep + 1}: reward={total_reward:.1f}")
    env.close()

    logger.info(
        f"Final: mean={np.mean(final_rewards):.1f} +/- {np.std(final_rewards):.1f} "
        f"over 10 episodes"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train controller with CMA-ES.")
    parser.add_argument("--vae_checkpoint", type=str, default="checkpoints/vae/best.pt")
    parser.add_argument("--mdn_rnn_checkpoint", type=str, default="checkpoints/mdn_rnn/best.pt")
    parser.add_argument("--save_dir", type=str, default="checkpoints/controller")
    parser.add_argument("--pop_size", type=int, default=32)
    parser.add_argument("--sigma_init", type=float, default=0.1)
    parser.add_argument("--max_generations", type=int, default=200)
    parser.add_argument("--target_reward", type=float, default=800.0)
    parser.add_argument("--num_rollouts", type=int, default=3)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_controller(
        vae_checkpoint=args.vae_checkpoint,
        mdn_rnn_checkpoint=args.mdn_rnn_checkpoint,
        save_dir=args.save_dir,
        pop_size=args.pop_size,
        sigma_init=args.sigma_init,
        max_generations=args.max_generations,
        target_reward=args.target_reward,
        num_rollouts=args.num_rollouts,
        max_steps=args.max_steps,
        seed=args.seed,
    )
