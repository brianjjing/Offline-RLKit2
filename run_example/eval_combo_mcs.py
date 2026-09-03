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
import random
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
from cost_func import (
    compute_acp_cost_model,
    weaning_score_model,
    weaning_score_model_gradient,
)


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


def _test_window_idxs(env, episodes, seed):
    """Episode-start indices covering the held-out test split, each at most once.

    rl_env._get_next_episode_start indexes train+val+test as one flat range, so the
    test split is [len(train)+len(val), total-max_steps]. Deterministic policy plus
    deterministic twin means one start window == one fixed episode, so sampling with
    replacement would just re-run the same episodes: enumerate instead, and shuffle
    before truncating so an --eval_episodes cap is not a temporal prefix.
    """
    wm = env.world_model
    lo = len(wm.data_train) + len(wm.data_val)
    idxs = list(range(lo, lo + len(wm.data_test) - env.max_steps + 1))
    random.Random(seed).shuffle(idxs)
    return idxs[:episodes]


def _evaluate(policy, eval_env, episodes, seed):
    """Sum max_steps rewards per episode, then mean/std across episodes.

    Scores only the held-out test split (see _test_window_idxs) -- the training
    scripts drop wm.data_test from the offline buffer, so these windows are unseen.

    Also accumulates the clinical episode metrics GORMPO reports in
    cormpo/helpers/evaluate.py::_evaluate_abiomed -- ACP and weaning score. Both
    are episode-level (they need the whole action sequence), so they are computed
    at terminal from the states the actions were taken in plus env.episode_actions
    (unnormalized p-levels, filled by AbiomedRLEnv.step).
    """
    policy.eval()
    returns, lengths, acps, ws_grad, ws_thr = [], [], [], [], []
    idxs = _test_window_idxs(eval_env, episodes, seed)

    for n, idx in enumerate(idxs, 1):
        obs = eval_env.reset(idx=idx)
        episode_reward, ep_states = 0.0, []

        # fixed horizon: termination_fn_abiomed never fires early, so max_steps is the
        # whole episode. The mean_length assert in __main__ is the check on that.
        for _ in range(eval_env.max_steps):
            action = policy.select_action(obs.reshape(1, -1), deterministic=True)
            ep_states.append(obs)                   # state this action was taken in
            obs, reward, _terminal, _ = eval_env.step(action.flatten())
            episode_reward += reward

        # read episode_actions BEFORE the next reset() clears it
        S, A = np.array(ep_states), eval_env.episode_actions
        acps.append(compute_acp_cost_model(eval_env.world_model, A, S))
        ws_grad.append(weaning_score_model_gradient(eval_env.world_model, S, A)[0])
        ws_thr.append(weaning_score_model(eval_env.world_model, S, A))

        returns.append(episode_reward)
        lengths.append(eval_env.max_steps)
        if n % 100 == 0:
            print("  %5d/%d  running mean %+.4f"
                  % (n, len(idxs), np.mean(returns)), flush=True)

    R, L = np.array(returns), np.array(lengths)
    ACP, WS, WST = np.array(acps), np.array(ws_grad), np.array(ws_thr)
    return {
        "n_episodes": len(R),
        "mean_return": R.mean(), "std_return": R.std(),
        "sem_return": R.std() / np.sqrt(len(R)),
        "mean_length": L.mean(), "std_length": L.std(),
        "mean_acp": ACP.mean(), "std_acp": ACP.std(), "max_acp": ACP.max(), "min_acp": ACP.min(),
        "mean_ws": WS.mean(), "std_ws": WS.std(), "max_ws": WS.max(), "min_ws": WS.min(),
        "mean_ws_thr": WST.mean(),
    }


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--policy-path", type=str, required=True)
    p.add_argument("--eval_episodes", type=int, default=1000,
                   help="GORMPO reports at 1000 (cormpo/mopo.py:200); 10 carries ~+/-1.3 of noise")
    # nargs so one invocation can sweep eval seeds, matching GORMPO's
    # helpers/evaluate.py --seeds. "--seed 42" still works (-> [42]), which is what
    # the mult_seed bash scripts pass.
    p.add_argument("--seed", "--seeds", type=int, nargs="+", default=[1],
                   help="eval seed(s): picks which test windows and their order")
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
                   default="/public/gormpo/10min_1hr_all_data.pkl")
    p.add_argument("--max_steps", type=int, default=6)
    return p.parse_args()


if __name__ == "__main__":
    args = get_args()
    seeds, all_res = args.seed, []

    for seed in seeds:
        args.seed = seed          # get_env seeds the twin with it
        np.random.seed(seed)
        torch.manual_seed(seed)

        env = get_env(args)
        policy = get_combo(env, args)
        res = _evaluate(policy, AbiomedGymCompat(env), args.eval_episodes, seed)

        print("\n---------------------------------------")
        print(f"Eval seed {seed}: {res['n_episodes']} held-out test episodes "
              f"(requested {args.eval_episodes}):")
        print(f"  Return:  {res['mean_return']:.4f} +/- {res['std_return']:.4f}")
        print(f"  SEM:     {res['sem_return']:.4f}   <- error bar on the mean")
        print(f"  Length:  {res['mean_length']:.1f} +/- {res['std_length']:.1f}")
        print(f"  ACP:     {res['mean_acp']:.4f} +/- {res['std_acp']:.4f}"
              f"   (max {res['max_acp']:.4f}, min {res['min_acp']:.4f})")
        print(f"  WS:      {res['mean_ws']:.5f} +/- {res['std_ws']:.5f}"
              f"   (max {res['max_ws']:.5f}, min {res['min_ws']:.5f})   <- gradient stability")
        print(f"  WS thr:  {res['mean_ws_thr']:.5f}   <- threshold stability (is_stable)")
        print("  Raw MCS scale, no x100 (cf. mb_policy_trainer.py:100-101).")
        print("---------------------------------------")

        # std is patient spread, not estimator error -- it must NOT shrink with episodes,
        # and a 6-step return cannot leave 6*[-2.0, +0.6393].
        assert -12.0 <= res["mean_return"] <= 6 * 0.6393, res["mean_return"]
        assert res["mean_length"] == args.max_steps, res["mean_length"]
        # WS is a per-stable-hour average of p-level deltas; ACP only sums |da| > 2, never negative.
        assert -1.0 <= res["mean_ws"] <= 2.0, res["mean_ws"]
        assert res["min_acp"] >= 0.0, res["min_acp"]

        # machine-readable row for the mult_seed bash scripts, emitted only once the
        # sanity asserts above have passed. Field order is fixed: combo_mcs.sh prepends
        # the TRAINING seed to it, so this must stay <n_episodes,return,sem>.
        print(f"CSVROW,{res['n_episodes']},{res['mean_return']:.6f},{res['sem_return']:.6f}")
        all_res.append((seed, res))

    if len(all_res) > 1:
        import statistics as st
        def agg(k):
            v = [r[k] for _, r in all_res]
            return st.mean(v), st.stdev(v)
        ret_m, ret_s = agg("mean_return")
        acp_m, acp_s = agg("mean_acp")
        ws_m, ws_s = agg("mean_ws")
        print("\n=======================================")
        print(f"Across {len(all_res)} eval seeds  ({args.policy_path})")
        for s, r in all_res:
            print(f"  seed {s:>3}: return {r['mean_return']:+.4f} (sem {r['sem_return']:.4f})"
                  f"  acp {r['mean_acp']:.4f}  ws {r['mean_ws']:.5f}")
        # +/- is spread over which test windows were drawn -- every eval seed scores the
        # SAME policy, so it is eval noise, NOT the across-training-seed error bar.
        print(f"  return: {ret_m:+.4f} +/- {ret_s:.4f}")
        print(f"  acp:    {acp_m:.4f} +/- {acp_s:.4f}")
        print(f"  ws:     {ws_m:.5f} +/- {ws_s:.5f}")
        print("=======================================")
        print(f"SEEDMEAN,{ret_m:.6f},{ret_s:.6f},{acp_m:.6f},{acp_s:.6f},{ws_m:.6f},{ws_s:.6f}")
