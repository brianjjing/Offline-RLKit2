"""Standalone evaluation of a trained COMBO(+DBG) policy on a D4RL mujoco task.

Mirrors eval_combo_mcs.py's trick: COMBOPolicy's saved state_dict only holds
{actor, critic1, critic1_old, critic2, critic2_old} (dynamics/cql_log_alpha
are not registered nn.Parameters), so a plain SACPolicy loads it strictly and
exposes the same .eval() / .select_action() surface COMBOPolicy would -- no
need to rebuild the dynamics ensemble or guardian just to run the actor.

    python run_example/eval_combo_d4rl.py \
        --policy-path "log/halfcheetah-medium-expert-v2/combo/<run>_sparse/model/policy.pth" \
        --task halfcheetah-medium-expert-v2 --eval_episodes 1000
"""
import argparse

import d4rl  # noqa: F401 -- registers get_normalized_score on the gym env
import gym
import numpy as np
import torch

from offlinerlkit.modules import ActorProb, Critic, TanhDiagGaussian
from offlinerlkit.nets import MLP
from offlinerlkit.policy import SACPolicy


def get_policy(env, args):
    obs_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))
    max_action = env.action_space.high[0]

    actor_backbone = MLP(input_dim=obs_dim, hidden_dims=args.hidden_dims)
    critic1_backbone = MLP(input_dim=obs_dim + action_dim, hidden_dims=args.hidden_dims)
    critic2_backbone = MLP(input_dim=obs_dim + action_dim, hidden_dims=args.hidden_dims)
    dist = TanhDiagGaussian(
        latent_dim=getattr(actor_backbone, "output_dim"),
        output_dim=action_dim,
        unbounded=True,
        conditioned_sigma=True,
        max_mu=max_action,
    )
    actor = ActorProb(actor_backbone, dist, args.device)
    critic1 = Critic(critic1_backbone, args.device)
    critic2 = Critic(critic2_backbone, args.device)

    policy = SACPolicy(
        actor, critic1, critic2,
        torch.optim.Adam(actor.parameters(), lr=args.actor_lr),
        torch.optim.Adam(critic1.parameters(), lr=args.critic_lr),
        torch.optim.Adam(critic2.parameters(), lr=args.critic_lr),
        tau=args.tau, gamma=args.gamma, alpha=args.alpha,
    )
    state_dict = torch.load(args.policy_path, map_location=args.device)
    policy.load_state_dict(state_dict)  # strict: catches hidden_dims/max_action drift
    return policy


def evaluate(policy, env, episodes, seed):
    """Same loop as MBPolicyTrainer._evaluate(), just with an episode cap of our choosing."""
    policy.eval()
    env.seed(seed)
    returns, lengths = [], []
    obs = env.reset()
    ep_reward, ep_length = 0.0, 0
    while len(returns) < episodes:
        action = policy.select_action(obs.reshape(1, -1), deterministic=True)
        obs, reward, terminal, _ = env.step(action.flatten())
        ep_reward += reward
        ep_length += 1
        if terminal:
            returns.append(ep_reward)
            lengths.append(ep_length)
            ep_reward, ep_length = 0.0, 0
            obs = env.reset()
    return np.array(returns), np.array(lengths)


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--policy-path", type=str, required=True)
    p.add_argument("--task", type=str, required=True)
    p.add_argument("--eval_episodes", type=int, default=1000,
                   help="COMBO+DBG training used 10; GORMPO-style reporting uses 1000.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cpu",
                   help="Eval is one obs at a time through a tiny MLP -- CPU avoids "
                        "CUDA-init overhead and GPU contention from other jobs.")

    # must match the values the checkpoint was trained with (mb_policy_trainer.py's
    # hidden_dims default / run_combo_dbg_sparse_d4rl.py args)
    p.add_argument("--hidden-dims", type=int, nargs="*", default=[256, 256, 256])
    p.add_argument("--actor-lr", type=float, default=1e-4)
    p.add_argument("--critic-lr", type=float, default=3e-4)
    p.add_argument("--tau", type=float, default=0.005)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--alpha", type=float, default=0.2)
    return p.parse_args()


if __name__ == "__main__":
    args = get_args()
    torch.set_num_threads(1)  # batch-size-1 eval gains nothing from multi-threading;
                              # keeps 15-way-parallel runs from thrashing a shared box
    env = gym.make(args.task)
    policy = get_policy(env, args)
    returns, lengths = evaluate(policy, env, args.eval_episodes, args.seed)

    norm_mean = env.get_normalized_score(returns.mean()) * 100
    norm_std = env.get_normalized_score(returns.std()) * 100

    print(f"policy: {args.policy_path}")
    print(f"episodes: {len(returns)}  length: {lengths.mean():.1f} +/- {lengths.std():.1f}")
    print(f"return: {returns.mean():.2f} +/- {returns.std():.2f}")
    print(f"normalized_score: {norm_mean:.2f} +/- {norm_std:.2f}")
    print(f"CSVROW,{args.task},{len(returns)},{norm_mean:.6f},{norm_std:.6f}")
