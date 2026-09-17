import argparse
import os
import sys
import random

import gym

import numpy as np
import torch


from offlinerlkit.nets import MLP
from offlinerlkit.modules import ActorProb, Critic, TanhDiagGaussian, EnsembleDynamicsModel
from offlinerlkit.dynamics import EnsembleDynamics
from offlinerlkit.utils.scaler import StandardScaler
from offlinerlkit.utils.termination_fns import get_termination_fn
from offlinerlkit.buffer import ReplayBuffer
from offlinerlkit.utils.logger import Logger, make_log_dirs
from offlinerlkit.policy_trainer import MBPolicyTrainer
from offlinerlkit.policy import COMBOPolicy

from offlinerlkit.utils.guardians import GUARDIAN_TYPES, load_guardian

sys.path.insert(0, "/home/brian/repos/OfflineRL-Kit2/abiomed_env")

from rl_env import AbiomedRLEnvFactory


"""
suggested hypers

halfcheetah-medium-v2: rollout-length=5, cql-weight=0.5
hopper-medium-v2: rollout-length=5, cql-weight=5.0
walker2d-medium-v2: rollout-length=1, cql-weight=5.0
halfcheetah-medium-replay-v2: rollout-length=5, cql-weight=0.5
hopper-medium-replay-v2: rollout-length=5, cql-weight=0.5
walker2d-medium-replay-v2: rollout-length=1, cql-weight=0.5
halfcheetah-medium-expert-v2: rollout-length=5, cql-weight=5.0
hopper-medium-expert-v2: rollout-length=5, cql-weight=5.0
walker2d-medium-expert-v2: rollout-length=1, cql-weight=5.0
"""


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo-name", type=str, default="combo")
    parser.add_argument("--task", type=str, default="abiomed")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--actor-lr", type=float, default=1e-4)
    parser.add_argument("--critic-lr", type=float, default=3e-4)
    parser.add_argument("--hidden-dims", type=int, nargs='*', default=[256, 256, 256])
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--auto-alpha", default=True)
    parser.add_argument("--target-entropy", type=int, default=None)
    parser.add_argument("--alpha-lr", type=float, default=1e-4)

    parser.add_argument("--cql-weight", type=float, default=5.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-q-backup", type=bool, default=False)
    parser.add_argument("--deterministic-backup", type=bool, default=True)
    parser.add_argument("--with-lagrange", type=bool, default=False)
    parser.add_argument("--lagrange-threshold", type=float, default=10.0)
    parser.add_argument("--cql-alpha-lr", type=float, default=3e-4)
    parser.add_argument("--num-repeat-actions", type=int, default=10)
    parser.add_argument("--uniform-rollout", type=bool, default=False)
    parser.add_argument("--rho-s", type=str, default="mix", choices=["model", "mix"])

    parser.add_argument("--dynamics-lr", type=float, default=1e-3)
    parser.add_argument("--dynamics-hidden-dims", type=int, nargs='*', default=[200, 200, 200, 200])
    parser.add_argument("--dynamics-weight-decay", type=float, nargs='*', default=[2.5e-5, 5e-5, 7.5e-5, 7.5e-5, 1e-4])
    parser.add_argument("--n-ensemble", type=int, default=7)
    parser.add_argument("--n-elites", type=int, default=5)
    parser.add_argument("--rollout-freq", type=int, default=1000)
    parser.add_argument("--rollout-batch-size", type=int, default=50000)
    parser.add_argument("--rollout-length", type=int, default=6)  # abiomed: full 6-hour episode (GORMPO used 5)
    parser.add_argument("--model-retain-epochs", type=int, default=5)
    parser.add_argument("--real-ratio", type=float, default=0.5)
    parser.add_argument("--load-dynamics-path", type=str, default=None)

    # vvv density guardian (DBG) args vvv
    parser.add_argument("--classifier-path", type=str, default=None,
                        help="Base path to the trained guardian checkpoint (no _model.pth/"
                             "_meta_data.pkl suffix), trained on [next_obs, action] of this task. Required.")
    parser.add_argument("--guardian-type", type=str, default=None, choices=list(GUARDIAN_TYPES),
                        help="Density estimator behind the guardian. Inferred from the "
                             "checkpoint basename when omitted (realnvp_42 -> realnvp).")
    parser.add_argument("--gormpo-root", type=str, default=None,
                        help="GORMPO checkout supplying the vae/kde/diffusion/neuralode "
                             "implementations. Defaults to $GORMPO_ROOT. Unused for realnvp.")
    parser.add_argument("--threshold-percentile", type=float, default=None,
                        help="Override thr with a pre-computed threshold_candidates entry.")
    parser.add_argument("--devid", type=int, default=0, help="GPU id for the KDE faiss index.")
    parser.add_argument("--guardian-chunk-size", type=int, default=4096,
                        help="Max samples per guardian score_samples call. Only binds for "
                             "neuralode, whose ODE solve allocates with the batch.")
    parser.add_argument("--penalty-coef", type=float, default=1.0,
                        help="OOD reward-penalty scale. MUST be nonzero or the guardian no-ops; tune per task.")
    parser.add_argument("--penalty-type", type=str, default="linear",
                        choices=["linear", "inverse", "tanh", "tanh_reg", "softplus"])
    # ^^^ density guardian (DBG) args ^^^

    parser.add_argument("--epoch", type=int, default=1000)
    parser.add_argument("--step-per-epoch", type=int, default=1000)
    parser.add_argument("--eval_episodes", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    return parser.parse_args()

#ADDED FOR ABIOMED ENV
def termination_fn_abiomed(obs, act, next_obs):
    # fixed-horizon episodes: the twin never terminates mid-rollout
    return np.zeros((len(obs), 1), dtype=bool)


#vvvADDED FOR ABIOMED ENV COMPATIBILITY - RAW ABIOMED SCOREvvv
class AbiomedGymCompat(gym.Wrapper):
    """Old-gym shim for the verbatim gymnasium-style AbiomedRLEnv.

    MBPolicyTrainer expects obs = reset(), a 4-tuple step(), and get_normalized_score().
    AbiomedRLEnv gives (obs, info) / (obs, r, terminated, truncated, info) and no score.
    gym.Wrapper forwards observation_space/action_space/seed for free.
    """
    def reset(self, **kwargs):
        obs, _info = self.env.reset(**kwargs)
        return obs

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs, reward, terminated or truncated, info

    def get_normalized_score(self, score):
        # mb_policy_trainer.py:100-101 multiplies by 100 expecting a D4RL fraction.
        # Clinical data has no random/expert reference, so undo the scale and let
        # eval/normalized_episode_reward carry the raw MCS return.
        return score / 100
#^^^ADDED FOR ABIOMED ENV COMPATIBILITY - RAW ABIOMED SCORE^^^



def build_abiomed_dataset(env, timesteps=6, feat=12):
    """Clinical offline dataset from the twin's train/val/test splits (one hour = one transition).

    Mirrors GORMPO cormpo/common/buffer.py::load_dataset branch (b).
    """
    wm = env.world_model
    # data_test is held out: the policy must never train on the windows it is scored on.
    splits = [wm.data_train, wm.data_val]
    data   = torch.cat([torch.as_tensor(s.data)   for s in splits], dim=0).float()   # (N,6,12)
    pl     = torch.cat([torch.as_tensor(s.pl)     for s in splits], dim=0).float()   # (N,6)
    labels = torch.cat([torch.as_tensor(s.labels) for s in splits], dim=0).float()   # (N,66)

    obs = data.reshape(-1, timesteps * feat)                                          # (N,72)
    next_obs = torch.cat(
        [labels.reshape(-1, timesteps, feat - 1), pl.reshape(-1, timesteps, 1)], dim=2
    ).reshape(-1, timesteps * feat)                                                   # (N,72)

    # one scalar p-level per hour: majority vote over the 6 unnormalized steps, then renormalize
    
    pl_unnorm = np.asarray(wm.unnorm_pl(pl))
    pl_mode = np.array([np.bincount(np.rint(r).astype(int)).argmax() for r in pl_unnorm]).reshape(-1, 1)
    actions = np.asarray(wm.normalize_pl(torch.as_tensor(pl_mode, dtype=torch.float32))).reshape(-1, 1)

    # reward from the env's own function, per hour, on each (6,12) next state
    nb = next_obs.reshape(-1, timesteps, feat)
    rewards = np.array([env._compute_reward(nb[i]) for i in range(nb.shape[0])], dtype=np.float32)
    # ponytail: O(N) python loop over _compute_reward — fine for the small clinical splits; vectorize if N grows

    obs_np, next_np = obs.cpu().numpy().astype(np.float32), next_obs.cpu().numpy().astype(np.float32)
    assert obs_np.shape[1] == timesteps * feat and actions.shape[1] == 1
    assert len(obs_np) == len(next_np) == len(actions) == len(rewards)
    return {
        "observations": obs_np,
        "next_observations": next_np,
        "actions": actions.astype(np.float32),
        "rewards": rewards,
        # fixed-horizon, non-terminating (consistent with termination_fn_abiomed). GORMPO branch-b
        # sets 1-done=1.0 here, which would zero all TD bootstrapping — treated as a bug, using 0.
        "terminals": np.zeros(len(obs_np), dtype=np.float32),
    }


def train(args=get_args()):
    # create env and dataset
    #env = gym.make(args.task)

    # if 'hopper' in args.task or 'halfcheetah' in args.task or 'walker2d' in args.task:
    #     dataset = qlearning_dataset(env)
    # else:
    #     dataset = d4rl.qlearning_dataset(env)

    #ADDED FOR ABIOMED ENV
    env = AbiomedRLEnvFactory.create_env(
        model_name="10min_1hr_all_data",
        model_path="/home/brian/repos/OfflineRL-Kit2/abiomed_env/data/10min_1hr_all_data_model.pth",
        data_path ="/public/gormpo/10min_1hr_all_data.pkl",   # full MCS data (17,865 windows), not the 300-window sample
        max_steps=6, action_space_type="continuous",
        reward_type="smooth",
        normalize_rewards=True,
        seed=42,
        device=args.device)  # override config.py's hardcoded cuda:1 (invalid under CUDA_VISIBLE_DEVICES=1)
    dataset = build_abiomed_dataset(env)
    
    args.obs_shape = env.observation_space.shape
    args.action_dim = np.prod(env.action_space.shape)


    #ADDED FOR ABIOMED ENV

    #args.max_action = env.action_space.high[0]
    args.max_action = max(abs(env.action_space.low[0]), abs(env.action_space.high[0]))
    # ^^^ symmetric bound so TanhDiagGaussian(max_mu) can reach both p-level extremes


    # seed: training stochasticity varies with args.seed, the eval protocol does not.
    # random is rl_env._get_next_episode_start's only consumer -- pinning it to 42 (as
    # run_combo_in_mcs.py does) gives every seed identical eval windows.
    random.seed(42)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    #env.seed(args.seed)

    # create policy model
    actor_backbone = MLP(input_dim=np.prod(args.obs_shape), hidden_dims=args.hidden_dims)
    critic1_backbone = MLP(input_dim=np.prod(args.obs_shape) + args.action_dim, hidden_dims=args.hidden_dims)
    critic2_backbone = MLP(input_dim=np.prod(args.obs_shape) + args.action_dim, hidden_dims=args.hidden_dims)
    dist = TanhDiagGaussian(
        latent_dim=getattr(actor_backbone, "output_dim"),
        output_dim=args.action_dim,
        unbounded=True,
        conditioned_sigma=True,
        max_mu=args.max_action
    )
    actor = ActorProb(actor_backbone, dist, args.device)
    critic1 = Critic(critic1_backbone, args.device)
    critic2 = Critic(critic2_backbone, args.device)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
    critic1_optim = torch.optim.Adam(critic1.parameters(), lr=args.critic_lr)
    critic2_optim = torch.optim.Adam(critic2.parameters(), lr=args.critic_lr)

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(actor_optim, args.epoch)

    if args.auto_alpha:
        target_entropy = args.target_entropy if args.target_entropy \
            else -np.prod(env.action_space.shape)
        args.target_entropy = target_entropy
        log_alpha = torch.zeros(1, requires_grad=True, device=args.device)
        alpha_optim = torch.optim.Adam([log_alpha], lr=args.alpha_lr)
        alpha = (target_entropy, log_alpha, alpha_optim)
    else:
        alpha = args.alpha

    # create dynamics
    load_dynamics_model = True if args.load_dynamics_path else False
    dynamics_model = EnsembleDynamicsModel(
        obs_dim=np.prod(args.obs_shape),
        action_dim=args.action_dim,
        hidden_dims=args.dynamics_hidden_dims,
        num_ensemble=args.n_ensemble,
        num_elites=args.n_elites,
        weight_decays=args.dynamics_weight_decay,
        device=args.device
    )
    dynamics_optim = torch.optim.Adam(
        dynamics_model.parameters(),
        lr=args.dynamics_lr
    )
    scaler = StandardScaler()

    #ADDED FOR ABIOMED ENV
    #termination_fn = get_termination_fn(task=args.task)
    termination_fn = termination_fn_abiomed

    # density guardian (DBG): load the estimator and hand it to EnsembleDynamics.
    assert args.classifier_path is not None, \
        "run_combo_dbg_mcs.py requires --classifier-path (trained guardian checkpoint)"
    classifier = load_guardian(
        args.classifier_path,
        device=args.device,
        guardian_type=args.guardian_type,
        devid=args.devid,
        threshold_percentile=args.threshold_percentile,
        task=args.task,
        gormpo_root=args.gormpo_root,
        chunk_size=args.guardian_chunk_size,
        # abiomed guardians are trained on [next_obs, action] like the D4RL ones, but their
        # checkpoints don't self-describe this the same way -- neuralode's metadata has no
        # target_dim (and "abiomed" doesn't match its D4RL-env-name path inference), and
        # abiomed_vae's metadata lacks hidden_dims (its arch is [256,256], not the [256,128]
        # default). Both are derivable from this env, so pass them instead of patching the
        # checkpoints on shared storage.
        target_dim=int(np.prod(args.obs_shape) + args.action_dim),
        vae_hidden_dims=[256, 256] if args.guardian_type == "vae" else None,
    )  # -> {'model','thr','mean','std','name'}

    dynamics = EnsembleDynamics(
        dynamics_model,
        dynamics_optim,
        scaler,
        termination_fn,
        classifier=classifier,
        penalty_coef=args.penalty_coef,
        penalty_type=args.penalty_type,
    )

    if args.load_dynamics_path:
        dynamics.load(args.load_dynamics_path)

    # create policy
    policy = COMBOPolicy(
        dynamics,
        actor,
        critic1,
        critic2,
        actor_optim,
        critic1_optim,
        critic2_optim,
        action_space=env.action_space,
        tau=args.tau,
        gamma=args.gamma,
        alpha=alpha,
        cql_weight=args.cql_weight,
        temperature=args.temperature,
        max_q_backup=args.max_q_backup,
        deterministic_backup=args.deterministic_backup,
        with_lagrange=args.with_lagrange,
        lagrange_threshold=args.lagrange_threshold,
        cql_alpha_lr=args.cql_alpha_lr,
        num_repeart_actions=args.num_repeat_actions,
        uniform_rollout=args.uniform_rollout,
        rho_s=args.rho_s
    )

    # create buffer
    real_buffer = ReplayBuffer(
        buffer_size=len(dataset["observations"]),
        obs_shape=args.obs_shape,
        obs_dtype=np.float32,
        action_dim=args.action_dim,
        action_dtype=np.float32,
        device=args.device
    )
    real_buffer.load_dataset(dataset)
    fake_buffer = ReplayBuffer(
        buffer_size=args.rollout_batch_size*args.rollout_length*args.model_retain_epochs,
        obs_shape=args.obs_shape,
        obs_dtype=np.float32,
        action_dim=args.action_dim,
        action_dtype=np.float32,
        device=args.device
    )

    # log
    # guardian_type in the dir name: running multiple estimators concurrently (see
    # combo_dbg_all_estimators_mcs.sh) means the plain seed+timestamp path isn't enough to
    # tell two runs apart if their timestamps land in the same second.
    log_dirs = make_log_dirs(args.task, args.algo_name, args.seed, vars(args), record_params=["guardian_type"])
    # key: output file name, value: output handler type
    output_config = {
        "consoleout_backup": "stdout",
        "policy_training_progress": "csv",
        "dynamics_training_progress": "csv",
        "tb": "tensorboard"
    }
    logger = Logger(log_dirs, output_config)
    logger.log_hyperparameters(vars(args))

    # create policy trainer
    policy_trainer = MBPolicyTrainer(
        policy=policy,
        #eval_env=env,
        eval_env=AbiomedGymCompat(env),
        real_buffer=real_buffer,
        fake_buffer=fake_buffer,
        logger=logger,
        rollout_setting=(args.rollout_freq, args.rollout_batch_size, args.rollout_length),
        epoch=args.epoch,
        step_per_epoch=args.step_per_epoch,
        batch_size=args.batch_size,
        real_ratio=args.real_ratio,
        eval_episodes=args.eval_episodes,
        lr_scheduler=lr_scheduler
    )

    # train
    if not load_dynamics_model:
        dynamics.train(real_buffer.sample_all(), logger, max_epochs_since_update=5)

    policy_trainer.train()


if __name__ == "__main__":
    train()