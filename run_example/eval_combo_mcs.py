"""Standalone evaluation of a trained COMBO policy on the Abiomed MCS twin.

COMBO equivalent of GORMPO's cormpo/helpers/evaluate.py: get_combo() mirrors
get_mopo() (build the skeleton, load the checkpoint, return the policy), and
__main__ mirrors its evaluation loop.

    python run_example/eval_combo_mcs.py \
        --policy-path "log/abiomed/combo/<run>/model/policy.pth" \
        --eval_episodes 1000 --device cuda:0
"""
import argparse
import os
import sys

import gym
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "abiomed_env")))

from offlinerlkit.nets import MLP
from offlinerlkit.modules import ActorProb, Critic, TanhDiagGaussian
from offlinerlkit.policy import SACPolicy

from rl_env import AbiomedRLEnvFactory


# Duplicated from run_combo_in_mcs.py:95 rather than imported: that module runs
# get_args() at import time (def train(args=get_args())), which eats our argv.
class AbiomedGymCompat(gym.Wrapper):
    """Old-gym shim for the gymnasium-style AbiomedRLEnv."""
    def reset(self, **kwargs):
        obs, _info = self.env.reset(**kwargs)
        return obs

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs, reward, terminated or truncated, info


def get_env(args):
    """Mirror of cormpo/helpers/evaluate.py:get_env."""
    env = AbiomedRLEnvFactory.create_env(
        model_name=args.model_name,
        model_path=args.model_path_wm,
        data_path=args.data_path_wm,
        max_steps=args.max_steps,
        action_space_type="continuous",
        reward_type="smooth",
        normalize_rewards=True,
        seed=args.seed,
        device=args.device,
    )
    args.obs_shape = env.observation_space.shape
    args.action_dim = int(np.prod(env.action_space.shape))
    # symmetric bound, matching run_combo_in_mcs.py:183 -- must equal the value used
    # at train time or the loaded actor's actions are silently rescaled.
    args.max_action = max(abs(env.action_space.low[0]), abs(env.action_space.high[0]))
    return env


def get_combo(env, args):
    """Initialize and load a trained COMBO policy. Mirror of get_mopo().

    COMBOPolicy's saved state_dict holds only {actor, critic1, critic1_old,
    critic2, critic2_old} -- cql_log_alpha is a bare tensor, not a registered
    parameter -- so a plain SACPolicy loads it strictly and exposes the same
    .eval() / .select_action() surface COMBOPolicy would.

    Returns:
        SACPolicy: policy with the checkpoint's weights restored.
    """
    obs_dim = int(np.prod(args.obs_shape))
    actor_backbone = MLP(input_dim=obs_dim, hidden_dims=args.hidden_dims)
    critic1_backbone = MLP(input_dim=obs_dim + args.action_dim, hidden_dims=args.hidden_dims)
    critic2_backbone = MLP(input_dim=obs_dim + args.action_dim, hidden_dims=args.hidden_dims)
    dist = TanhDiagGaussian(
        latent_dim=getattr(actor_backbone, "output_dim"),
        output_dim=args.action_dim,
        unbounded=True,
        conditioned_sigma=True,
        max_mu=args.max_action,
    )

    actor = ActorProb(actor_backbone, dist, args.device)
    critic1 = Critic(critic1_backbone, args.device)
    critic2 = Critic(critic2_backbone, args.device)

    policy = SACPolicy(
        actor,
        critic1,
        critic2,
        torch.optim.Adam(actor.parameters(), lr=args.actor_lr),
        torch.optim.Adam(critic1.parameters(), lr=args.critic_lr),
        torch.optim.Adam(critic2.parameters(), lr=args.critic_lr),
        tau=args.tau,
        gamma=args.gamma,
        alpha=args.alpha,
    )
    policy_state_dict = torch.load(args.policy_path, map_location=args.device)
    policy.load_state_dict(policy_state_dict)  # strict: catches hidden_dims/max_action drift
    return policy


def _evaluate(policy, eval_env, episodes):
    """Identical arithmetic to MBPolicyTrainer._evaluate: sum max_steps rewards
    per episode, then mean/std across episodes."""
    policy.eval()
    returns, lengths = [], []
    obs = eval_env.reset()
    episode_reward, episode_length = 0.0, 0

    while len(returns) < episodes:
        action = policy.select_action(obs.reshape(1, -1), deterministic=True)
        obs, reward, terminal, _ = eval_env.step(action.flatten())
        episode_reward += reward
        episode_length += 1

        if terminal:
            returns.append(episode_reward)
            lengths.append(episode_length)
            episode_reward, episode_length = 0.0, 0
            obs = eval_env.reset()
            if len(returns) % 100 == 0:
                print("  %5d/%d  running mean %+.4f"
                      % (len(returns), episodes, np.mean(returns)), flush=True)

    R, L = np.array(returns), np.array(lengths)
    return {
        "mean_return": R.mean(), "std_return": R.std(),
        "sem_return": R.std() / np.sqrt(len(R)),
        "mean_length": L.mean(), "std_length": L.std(),
    }


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--policy-path", type=str, required=True)
    p.add_argument("--eval_episodes", type=int, default=1000,
                   help="GORMPO reports at 1000 (cormpo/mopo.py:200); 10 carries ~+/-1.3 of noise")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # must match the values the checkpoint was trained with
    p.add_argument("--hidden-dims", type=int, nargs="*", default=[256, 256, 256])
    p.add_argument("--actor-lr", type=float, default=1e-4)
    p.add_argument("--critic-lr", type=float, default=3e-4)
    p.add_argument("--tau", type=float, default=0.005)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--alpha", type=float, default=0.2)

    p.add_argument("--model_name", type=str, default="10min_1hr_all_data")
    p.add_argument("--model_path_wm", type=str,
                   default="/home/brian/repos/OfflineRL-Kit2/abiomed_env/data/10min_1hr_all_data_model.pth")
    p.add_argument("--data_path_wm", type=str,
                   default="/home/brian/repos/OfflineRL-Kit2/abiomed_env/data/10min_1hr_all_data.pkl")
    p.add_argument("--max_steps", type=int, default=6)
    return p.parse_args()


if __name__ == "__main__":
    args = get_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    env = get_env(args)
    policy = get_combo(env, args)
    res = _evaluate(policy, AbiomedGymCompat(env), args.eval_episodes)

    print("\n---------------------------------------")
    print(f"Evaluation over {args.eval_episodes} episodes:")
    print(f"  Return:  {res['mean_return']:.4f} +/- {res['std_return']:.4f}")
    print(f"  SEM:     {res['sem_return']:.4f}   <- error bar on the mean")
    print(f"  Length:  {res['mean_length']:.1f} +/- {res['std_length']:.1f}")
    print("  Raw MCS scale, no x100 (cf. mb_policy_trainer.py:100-101).")
    print("---------------------------------------")

    # std is patient spread, not estimator error -- it must NOT shrink with episodes,
    # and a 6-step return cannot leave 6*[-2.0, +0.6393].
    assert -12.0 <= res["mean_return"] <= 6 * 0.6393, res["mean_return"]
    assert res["mean_length"] == args.max_steps, res["mean_length"]
