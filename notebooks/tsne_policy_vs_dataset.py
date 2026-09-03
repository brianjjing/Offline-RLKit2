#!/usr/bin/env python3
"""t-SNE support-overlap plots for COMBO, ported from GORMPO/notebooks/tsne_policy_vs_dataset.py.

Same figure and CLI; only the policy side differs. COMBO checkpoints are
`state_dict()`s of a COMBOPolicy, so we rebuild just the actor (ActorProb +
TanhDiagGaussian, offlinerlkit) and load the `actor.*` keys. COMBO does not
normalize observations for the policy (the scaler is dynamics-only), so the
actor consumes raw obs exactly as MBPolicyTrainer._evaluate does.

Policies come from --log-root discovery (default) or a GORMPO-style
--policy-json / --policy label=path.
"""
import argparse
import json
import os
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

from offlinerlkit.nets import MLP
from offlinerlkit.modules import ActorProb, TanhDiagGaussian


# Guardian dir basename -> panel label. Order here is the panel order.
ESTIMATOR_LABELS = [
    (None, "COMBO"),
    ("kde", "COMBO-KDE"),
    ("vae", "COMBO-VAE"),
    ("realnvp", "COMBO-RealNVP"),
    ("diffusion", "COMBO-DDPM"),
    ("neuralode", "COMBO-NeuralODE"),
]


@dataclass
class PolicySpec:
    label: str
    path: str
    hidden_dims: Optional[List[int]] = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate t-SNE plots for offline dataset support vs COMBO policy support."
    )
    parser.add_argument("--task", type=str, default="halfcheetah-medium-expert-v2")
    parser.add_argument("--dataset-path", type=str, default=None)
    parser.add_argument(
        "--policy",
        action="append",
        default=[],
        help='Policy spec as "label=/abs/path/to/policy.pth". Can be repeated.',
    )
    parser.add_argument(
        "--policy-json",
        type=str,
        default=None,
        help="JSON file with list of {'label': ..., 'path': ...}.",
    )
    parser.add_argument(
        "--log-root",
        type=str,
        default=None,
        help="Scan <log-root>/<task>/combo/*_sparse for finished runs and label each "
             "by its density guardian. Used when no --policy/--policy-json is given.",
    )
    parser.add_argument(
        "--seed-filter",
        type=int,
        default=42,
        help="Only discover runs with this seed (-1 for all seeds).",
    )
    parser.add_argument("--n-rollout-episodes", type=int, default=100)
    parser.add_argument("--max-offline-samples", type=int, default=10000)
    parser.add_argument("--max-rollout-samples", type=int, default=5000000)
    parser.add_argument("--perplexity", type=float, default=40.0)
    parser.add_argument("--n-iter", type=int, default=1000)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--support-source",
        type=str,
        choices=["rollout", "offline_policy"],
        default="offline_policy",
        help="Use real env rollouts or infer policy support from offline states.",
    )
    parser.add_argument(
        "--env-id",
        type=str,
        default=None,
        help="Gym env id for rollout mode. Defaults to --task.",
    )
    parser.add_argument("--output-dir", type=str, default="results/tsne_policy_vs_dataset")
    parser.add_argument("--panel-cols", type=int, default=6)
    parser.add_argument("--save-pdf", action="store_true")
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--point-size", type=float, default=8.0)
    parser.add_argument("--offline-alpha", type=float, default=0.35)
    parser.add_argument("--policy-alpha", type=float, default=0.55)
    return parser.parse_args()


def clean_path(path: str) -> str:
    return path.strip().strip('"').strip("'").replace("\n", "").replace("\r", "")


def _guardian_label(classifier_path: Optional[str]) -> Optional[str]:
    """Map a --classifier-path (e.g. .../halfcheetah_..._sparse_3/realnvp_42) to a panel label."""
    if not classifier_path:
        return "COMBO"
    stem = Path(clean_path(classifier_path)).name.lower()
    # Guardians are saved as <type> or <type>_<seed> (realnvp_42, kde_123, diffusion_42).
    for key, label in ESTIMATOR_LABELS:
        if key is not None and stem.startswith(key):
            return label
    return None


def discover_policies(log_root: str, task: str, seed_filter: int) -> List[PolicySpec]:
    """Collect finished COMBO sparse runs, one per density estimator, newest first."""
    run_dirs = sorted(Path(log_root, task, "combo").glob("*_sparse"))
    by_label: Dict[str, PolicySpec] = {}
    for run in run_dirs:
        ckpt = run / "model" / "policy.pth"
        hp_path = run / "record" / "hyper_param.json"
        if not ckpt.exists() or not hp_path.exists():
            continue  # unfinished run
        with open(hp_path, "r", encoding="utf-8") as f:
            hp = json.load(f)
        if seed_filter >= 0 and hp.get("seed") != seed_filter:
            continue
        label = _guardian_label(hp.get("classifier_path"))
        if label is None:
            print(f"[discover] unknown guardian for {run.name}: {hp.get('classifier_path')}")
            continue
        # sorted() is chronological (timestamps in the dir name), so this keeps the latest.
        by_label[label] = PolicySpec(label, str(ckpt), hp.get("hidden_dims"))

    specs = [by_label[lab] for _, lab in ESTIMATOR_LABELS if lab in by_label]
    missing = [lab for _, lab in ESTIMATOR_LABELS if lab not in by_label]
    print(f"[discover] {task}: found {[s.label for s in specs]}")
    if missing:
        print(f"[discover] {task}: no finished checkpoint for {missing} -- these panels are omitted.")
    return specs


def parse_policy_specs(args: argparse.Namespace) -> List[PolicySpec]:
    specs: List[PolicySpec] = []

    if args.policy_json:
        with open(args.policy_json, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        for entry in loaded:
            specs.append(PolicySpec(entry["label"], clean_path(entry["path"]), entry.get("hidden_dims")))

    for raw in args.policy:
        if "=" not in raw:
            raise ValueError(f'Invalid --policy value "{raw}", expected "label=path"')
        label, path = raw.split("=", 1)
        specs.append(PolicySpec(label.strip(), clean_path(path)))

    if not specs and args.log_root:
        specs = discover_policies(args.log_root, args.task, args.seed_filter)

    if not specs:
        raise ValueError("No policies provided. Use --log-root, --policy-json, or --policy.")

    return specs


def apply_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 11,
            "figure.titlesize": 14,
            "axes.linewidth": 0.8,
            "savefig.bbox": "tight",
        }
    )


def _normalize_loaded_dataset(dataset_obj):
    if isinstance(dataset_obj, dict):
        return dataset_obj
    raise ValueError(f"Unsupported dataset object type: {type(dataset_obj)}")


def _load_dataset_file(dataset_path: str) -> dict:
    suffix = Path(dataset_path).suffix.lower()
    if suffix == ".npz":
        npz_data = np.load(dataset_path, allow_pickle=True)
        return {k: npz_data[k] for k in npz_data.files}
    with open(dataset_path, "rb") as f:
        return _normalize_loaded_dataset(pickle.load(f))


def _extract_key(dataset: dict, candidates: Sequence[str]):
    for key in candidates:
        if key in dataset:
            return dataset[key], key
    raise KeyError(f"None of keys found: {candidates}")


def load_dataset(task: str, dataset_path: Optional[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    import gym

    if dataset_path is not None and os.path.exists(dataset_path):
        dataset = _load_dataset_file(dataset_path)
        print(f"[dataset] Loaded sparse/custom dataset: {dataset_path}")
    else:
        import d4rl  # noqa: F401  # required to register D4RL env datasets

        env_tmp = gym.make(task)
        dataset = d4rl.qlearning_dataset(env_tmp)
        env_tmp.close()
        print(f"[dataset] Loaded D4RL dataset for task: {task}")

    obs, obs_key = _extract_key(dataset, ("observations", "obs", "states"))
    next_obs, next_obs_key = _extract_key(dataset, ("next_observations", "next_obs", "next_states"))
    actions, act_key = _extract_key(dataset, ("actions", "acts", "action"))
    print(f"[dataset] Using keys: obs='{obs_key}', next_obs='{next_obs_key}', actions='{act_key}'")
    return obs, next_obs, actions


def build_actor(
    obs_dim: int, action_dim: int, max_action: float, hidden_dims: Sequence[int], device: str
) -> ActorProb:
    """Mirror the actor built in run_combo_dbg_sparse_d4rl.py."""
    backbone = MLP(input_dim=obs_dim, hidden_dims=list(hidden_dims))
    dist = TanhDiagGaussian(
        latent_dim=backbone.output_dim,
        output_dim=action_dim,
        unbounded=True,
        conditioned_sigma=True,
        max_mu=max_action,
    )
    return ActorProb(backbone, dist, device)


def load_actor(actor: ActorProb, policy_path: str, device: str) -> None:
    """COMBO saves the whole policy state_dict; keep only the actor.* entries."""
    try:
        state_dict = torch.load(policy_path, map_location=device, weights_only=False)
    except TypeError:
        state_dict = torch.load(policy_path, map_location=device)
    actor_sd = {k[len("actor."):]: v for k, v in state_dict.items() if k.startswith("actor.")}
    if not actor_sd:
        raise KeyError(f"No 'actor.*' keys in checkpoint {policy_path}")
    actor.load_state_dict(actor_sd)
    actor.eval()


def select_action(actor: ActorProb, obs: np.ndarray, deterministic: bool) -> np.ndarray:
    """Same as SACPolicy.select_action, without needing the full policy object.

    The np.array(..., float64) copy is load-bearing: torch here is built against a
    different numpy ABI, so arrays handed back by `Tensor.numpy()` fail ufunc
    dispatch (mujoco's action rescaling raises a bare TypeError on them).
    """
    with torch.no_grad():
        dist = actor(obs)
        action, _ = dist.mode() if deterministic else dist.rsample()
    return np.array(action.cpu().numpy(), dtype=np.float64)


def collect_rollouts(
    actor: ActorProb, env, n_episodes: int, deterministic: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    all_next_obs: List[np.ndarray] = []
    all_actions: List[np.ndarray] = []

    for ep in range(n_episodes):
        obs = env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]
        done = False
        truncated = False

        while not done and not truncated:
            action = select_action(actor, obs.reshape(1, -1), deterministic).flatten()
            step_out = env.step(action)

            if len(step_out) == 5:
                next_obs, _, done, truncated, _ = step_out
            else:
                next_obs, _, done, _ = step_out
                truncated = False

            all_next_obs.append(next_obs.copy())
            all_actions.append(np.asarray(action).copy())
            obs = next_obs

        if (ep + 1) % 10 == 0:
            print(f"[rollout] finished episode {ep + 1}/{n_episodes}")

    return np.asarray(all_next_obs), np.asarray(all_actions)


def collect_policy_support_from_offline(
    actor: ActorProb,
    offline_obs: np.ndarray,
    offline_next_obs: np.ndarray,
    max_samples: int,
    deterministic: bool,
    random_seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(random_seed)
    n = offline_obs.shape[0]
    k = min(max_samples, n)
    idx = rng.choice(n, size=k, replace=False)
    actions = select_action(actor, offline_obs[idx], deterministic)
    return offline_next_obs[idx], np.asarray(actions)


def prepare_tsne_input(
    offline_next_obs: np.ndarray,
    offline_actions: np.ndarray,
    rollout_next_obs: np.ndarray,
    rollout_actions: np.ndarray,
    max_offline_samples: int,
    max_rollout_samples: int,
    random_seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(random_seed)

    n_off = min(max_offline_samples, offline_next_obs.shape[0])
    off_idx = rng.choice(offline_next_obs.shape[0], size=n_off, replace=False)
    offline_sa = np.concatenate([offline_next_obs[off_idx], offline_actions[off_idx]], axis=1)

    if rollout_next_obs.shape[0] > max_rollout_samples:
        roll_idx = rng.choice(rollout_next_obs.shape[0], size=max_rollout_samples, replace=False)
        rollout_sa = np.concatenate([rollout_next_obs[roll_idx], rollout_actions[roll_idx]], axis=1)
    else:
        rollout_sa = np.concatenate([rollout_next_obs, rollout_actions], axis=1)

    all_sa = np.concatenate([offline_sa, rollout_sa], axis=0)
    labels = np.concatenate(
        [np.zeros(offline_sa.shape[0], dtype=int), np.ones(rollout_sa.shape[0], dtype=int)],
        axis=0,
    )
    print(
        f"[tsne] offline transitions used={offline_sa.shape[0]:,} "
        f"rollout transitions used={rollout_sa.shape[0]:,} "
        f"total={all_sa.shape[0]:,}"
    )
    return all_sa, labels


def run_tsne(all_sa: np.ndarray, labels: np.ndarray, perplexity: float, n_iter: int, random_seed: int):
    all_sa_scaled = StandardScaler().fit_transform(all_sa)

    print(
        f"[tsne] points={all_sa_scaled.shape[0]:,}, perplexity={perplexity}, n_iter={n_iter}, seed={random_seed}"
    )
    # sklearn changed TSNE arg name from n_iter -> max_iter in newer versions.
    tsne_kwargs = dict(
        n_components=2,
        perplexity=perplexity,
        random_state=random_seed,
        verbose=1,
        init="pca",
        learning_rate="auto",
    )
    try:
        tsne = TSNE(**tsne_kwargs, n_iter=n_iter)
    except TypeError:
        tsne = TSNE(**tsne_kwargs, max_iter=n_iter)
    emb = tsne.fit_transform(all_sa_scaled)
    return emb[labels == 0], emb[labels == 1]


def draw_panel(
    ax,
    emb_offline: np.ndarray,
    emb_rollout: np.ndarray,
    title: str,
    point_size: float,
    offline_alpha: float,
    policy_alpha: float,
):
    ax.scatter(
        emb_offline[:, 0],
        emb_offline[:, 1],
        c="steelblue",
        alpha=offline_alpha,
        s=point_size,
        label="Offline dataset",
        linewidths=0,
    )
    ax.scatter(
        emb_rollout[:, 0],
        emb_rollout[:, 1],
        c="tomato",
        alpha=policy_alpha,
        s=point_size * 1.5,
        label="Policy rollout",
        linewidths=0,
    )
    ax.set_title(title, fontsize=13)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")


def save_single_figure(
    out_path: Path,
    emb_offline: np.ndarray,
    emb_rollout: np.ndarray,
    title: str,
    args: argparse.Namespace,
):
    fig, ax = plt.subplots(figsize=(10, 8))
    draw_panel(
        ax=ax,
        emb_offline=emb_offline,
        emb_rollout=emb_rollout,
        title=title,
        point_size=args.point_size,
        offline_alpha=args.offline_alpha,
        policy_alpha=args.policy_alpha,
    )
    ax.legend(markerscale=3, fontsize=11)
    plt.tight_layout()
    fig.savefig(out_path, dpi=args.dpi)
    if args.save_pdf:
        fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


def save_combined_figure(
    out_path: Path,
    task: str,
    model_results: Sequence[Tuple[str, np.ndarray, np.ndarray]],
    panel_cols: int,
    args: argparse.Namespace,
):
    n = len(model_results)
    ncols = max(1, min(panel_cols, n))
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(4.6 * ncols, 4.0 * nrows),
        squeeze=False,
    )

    flat_axes = axes.flatten()
    for i, (label, emb_off, emb_roll) in enumerate(model_results):
        draw_panel(
            ax=flat_axes[i],
            emb_offline=emb_off,
            emb_rollout=emb_roll,
            title=label,
            point_size=args.point_size,
            offline_alpha=args.offline_alpha,
            policy_alpha=args.policy_alpha,
        )

    for j in range(n, len(flat_axes)):
        flat_axes[j].axis("off")

    handles, labels = flat_axes[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    # GORMPO's original stacks the legend on top of the suptitle. Lay the axes out
    # first, then put title above legend above axes; savefig bbox="tight" keeps both.
    plt.tight_layout()
    fig.legend(
        by_label.values(),
        by_label.keys(),
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=2,
        markerscale=3,
        fontsize=11,
        frameon=True,
    )
    fig.suptitle(f"t-SNE Support Overlap: {task}", y=1.0 + 0.9 / fig.get_figheight(), fontsize=13)
    fig.savefig(out_path, dpi=args.dpi)
    if args.save_pdf:
        fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


def sanitize_filename(name: str) -> str:
    return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in name).strip("_")


def main() -> None:
    import gym

    args = parse_args()
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    apply_plot_style()

    specs = parse_policy_specs(args)
    output_dir = Path(args.output_dir) / sanitize_filename(args.task)
    output_dir.mkdir(parents=True, exist_ok=True)

    offline_obs, offline_next_obs, offline_actions = load_dataset(args.task, args.dataset_path)
    env = None
    obs_dim = offline_obs.shape[1]
    action_dim = offline_actions.shape[1]
    max_action = 1.0

    if args.support_source == "rollout":
        # D4RL registers env IDs on import; required even if dataset is loaded from file.
        # `mujoco_py` sometimes requires /usr/lib/nvidia to be present in LD_LIBRARY_PATH.
        nvidia_lib = "/usr/lib/nvidia"
        if os.path.isdir(nvidia_lib):
            ld = os.environ.get("LD_LIBRARY_PATH", "")
            if nvidia_lib not in ld.split(":"):
                os.environ["LD_LIBRARY_PATH"] = (ld + (":" if ld else "") + nvidia_lib).strip(":")

        import d4rl  # noqa: F401

        env = gym.make(args.env_id or args.task)
        obs_dim = env.observation_space.shape[0]
        action_dim = int(np.prod(env.action_space.shape))
        max_action = float(env.action_space.high[0])

    model_results: List[Tuple[str, np.ndarray, np.ndarray]] = []

    for spec in specs:
        policy_path = clean_path(spec.path)
        if not os.path.exists(policy_path) or os.path.getsize(policy_path) == 0:
            print(f"[skip] missing or empty policy checkpoint: {policy_path}")
            continue

        print(f"[policy] {spec.label} -> {policy_path}")
        actor = build_actor(
            obs_dim=obs_dim,
            action_dim=action_dim,
            max_action=max_action,
            hidden_dims=spec.hidden_dims or [256, 256, 256],
            device=args.device,
        )
        try:
            load_actor(actor, policy_path, args.device)
        except Exception as exc:
            print(f"[skip] failed to load policy '{spec.label}': {type(exc).__name__}: {exc}")
            continue

        if args.support_source == "rollout":
            rollout_next_obs, rollout_actions = collect_rollouts(
                actor=actor,
                env=env,
                n_episodes=args.n_rollout_episodes,
                deterministic=args.deterministic,
            )
        else:
            rollout_next_obs, rollout_actions = collect_policy_support_from_offline(
                actor=actor,
                offline_obs=offline_obs,
                offline_next_obs=offline_next_obs,
                max_samples=args.max_rollout_samples,
                deterministic=args.deterministic,
                random_seed=args.random_seed,
            )
        all_sa, labels = prepare_tsne_input(
            offline_next_obs=offline_next_obs,
            offline_actions=offline_actions,
            rollout_next_obs=rollout_next_obs,
            rollout_actions=rollout_actions,
            max_offline_samples=args.max_offline_samples,
            max_rollout_samples=args.max_rollout_samples,
            random_seed=args.random_seed,
        )
        emb_offline, emb_rollout = run_tsne(
            all_sa=all_sa,
            labels=labels,
            perplexity=args.perplexity,
            n_iter=args.n_iter,
            random_seed=args.random_seed,
        )

        single_path = output_dir / f"{sanitize_filename(spec.label)}_tsne.png"
        save_single_figure(
            out_path=single_path,
            emb_offline=emb_offline,
            emb_rollout=emb_rollout,
            title=f"{spec.label} ({args.task})",
            args=args,
        )
        print(f"[saved] {single_path}")
        model_results.append((spec.label, emb_offline, emb_rollout))

    if env is not None:
        env.close()

    if not model_results:
        raise RuntimeError("No figures generated. Check policy paths and environment setup.")

    panel_path = output_dir / "combined_panel.png"
    save_combined_figure(
        out_path=panel_path,
        task=args.task,
        model_results=model_results,
        panel_cols=args.panel_cols,
        args=args,
    )
    print(f"[saved] {panel_path}")

    manifest = {
        "task": args.task,
        "output_dir": str(output_dir),
        "num_models": len(model_results),
        "models": [{"label": m[0]} for m in model_results],
        "combined_panel": str(panel_path),
    }
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"[saved] {manifest_path}")


if __name__ == "__main__":
    main()
