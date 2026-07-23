#!/usr/bin/env python3
"""Evaluate random and SAC baselines on Abiomed; compute D4RL-style normalized score."""

import argparse
import os

import numpy as np
from stable_baselines3 import SAC

from rl_env import AbiomedRLEnvFactory


def make_env(model_name, seed, device, max_steps):
    return AbiomedRLEnvFactory.create_env(
        model_name=model_name,
        max_steps=max_steps,
        action_space_type="continuous",
        reward_type="smooth",
        normalize_rewards=True,
        seed=seed,
        device=device,
    )


def evaluate_episodes(env, num_episodes, seed, get_action):
    returns = []
    lengths = []
    for i in range(num_episodes):
        obs, _ = env.reset(seed=seed + i)
        total_reward = 0.0
        steps = 0
        for _ in range(env.max_steps):
            action = get_action(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            steps += 1
            if terminated or truncated:
                break
        returns.append(total_reward)
        lengths.append(steps)
    return {
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "returns": returns,
        "mean_length": float(np.mean(lengths)),
    }


def d4rl_normalized_score(agent_return, random_return, expert_return):
    """Same formula as D4RL: (agent - random) / (expert - random) * 100."""
    return (agent_return - random_return) / (expert_return - random_return) * 100.0


def main():
    parser = argparse.ArgumentParser(description="Abiomed baselines + D4RL normalized score")
    parser.add_argument("--eval_episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--model_name", type=str, default="10min_1hr_all_data")
    parser.add_argument("--max_steps", type=int, default=50, help="Must match LEQ eval (default 50)")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--sac_path",
        type=str,
        default="data/sac_20250827_0837.zip",
        help="SAC checkpoint used as expert upper bound",
    )
    parser.add_argument(
        "--leq_return",
        type=float,
        default=11.805463917348188,
        help="LEQ mean eval return at step 1M",
    )
    parser.add_argument(
        "--expert_return",
        type=float,
        default=None,
        help="Manual expert return (skip SAC eval if set with --skip_sac)",
    )
    parser.add_argument("--skip_sac", action="store_true", help="Only evaluate random policy")
    args = parser.parse_args()

    env = make_env(args.model_name, args.seed, args.device, args.max_steps)
    random_stats = evaluate_episodes(
        env,
        args.eval_episodes,
        args.seed,
        lambda obs: env.action_space.sample(),
    )
    env.close()

    print(f"Settings: max_steps={args.max_steps}, episodes={args.eval_episodes}, seed={args.seed}")
    print(f"\nRandom policy (D4RL ref_min equivalent)")
    print(f"  Mean return: {random_stats['mean_return']:.3f} ± {random_stats['std_return']:.3f}")
    print(f"  Returns: {[round(r, 3) for r in random_stats['returns']]}")

    expert_return = args.expert_return
    if not args.skip_sac and expert_return is None:
        if not os.path.exists(args.sac_path):
            raise FileNotFoundError(
                f"SAC model not found: {args.sac_path}\n"
                "Use --skip_sac and pass --expert_return manually if needed."
            )
        sac_env = make_env(args.model_name, args.seed, args.device, args.max_steps)
        policy = SAC.load(args.sac_path, env=sac_env, device=args.device)

        def sac_action(obs):
            action, _ = policy.predict(obs, deterministic=False)
            return action

        sac_stats = evaluate_episodes(
            sac_env, args.eval_episodes, args.seed, sac_action
        )
        sac_env.close()
        expert_return = sac_stats["mean_return"]
        print(f"\nSAC expert policy (D4RL ref_max equivalent)")
        print(f"  Mean return: {sac_stats['mean_return']:.3f} ± {sac_stats['std_return']:.3f}")
        print(f"  Returns: {[round(r, 3) for r in sac_stats['returns']]}")

    if expert_return is None:
        print("\nProvide SAC checkpoint or --expert_return to compute normalized score.")
        return

    norm = d4rl_normalized_score(args.leq_return, random_stats["mean_return"], expert_return)
    print(f"\nLEQ agent return: {args.leq_return:.3f}")
    print(f"D4RL normalized score = ({args.leq_return:.3f} - {random_stats['mean_return']:.3f}) "
          f"/ ({expert_return:.3f} - {random_stats['mean_return']:.3f}) * 100")
    print(f"  = {norm:.1f}")


if __name__ == "__main__":
    main()
