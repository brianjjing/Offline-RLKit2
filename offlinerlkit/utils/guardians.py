"""Load any of GORMPO's density guardians for COMBO+DBG.

Ported from GORMPO/train.py:181-262. EnsembleDynamics only wants a dict of
{'model', 'thr', 'mean', 'std', 'name'} whose model exposes
``score_samples(np_array, device) -> log-probs``, so each branch here just has to
build that much.

Imports are per-branch on purpose. RealNVP resolves to this repo's own vendored,
dependency-free copy, so that path needs nothing extra installed and behaves exactly
as it did before this module existed. The other four are imported from the GORMPO
checkout and pull in its heavier deps (faiss, diffusers, torchdiffeq, sklearn, scipy,
seaborn, pyyaml) -- so you only pay for the estimator you actually ask for.
"""
import json
import os
import pickle
import sys
from pathlib import Path

GUARDIAN_TYPES = ("realnvp", "vae", "kde", "diffusion", "neuralode")

# GORMPO owns the VAE/KDE/diffusion/neuralODE implementations; we import rather than fork.
DEFAULT_GORMPO_ROOT = os.environ.get("GORMPO_ROOT", "/home/brian/repos/GORMPO")


def infer_guardian_type(classifier_path: str) -> str:
    """Map a checkpoint path to its estimator type.

    Matches on the basename, not the whole path, so a task directory that happens to
    contain a substring like 'vae' cannot hijack the dispatch. Guardians are saved as
    <type> or <type>_<seed> (realnvp_42, kde_123, diffusion_42).
    """
    stem = Path(classifier_path.rstrip("/")).name.lower()
    if stem.endswith(".pt") or stem.endswith(".pth"):
        stem = Path(classifier_path).parent.name.lower()
    for kind in GUARDIAN_TYPES:
        if stem.startswith(kind):
            return kind
    raise ValueError(
        f"Cannot infer guardian type from '{classifier_path}' (basename '{stem}'). "
        f"Pass --guardian-type explicitly; expected one of {GUARDIAN_TYPES}."
    )


def _use_gormpo(gormpo_root: str) -> str:
    root = gormpo_root or DEFAULT_GORMPO_ROOT
    if not os.path.isdir(root):
        raise FileNotFoundError(
            f"GORMPO checkout not found at '{root}'. Set GORMPO_ROOT or pass --gormpo-root; "
            "only the realnvp guardian can be loaded without it."
        )
    if root not in sys.path:
        sys.path.insert(0, root)
    return root


def load_guardian(
    classifier_path: str,
    device: str = "cpu",
    guardian_type: str = None,
    devid: int = 0,
    threshold_percentile=None,
    task: str = None,
    gormpo_root: str = None,
    chunk_size: int = 4096,
    target_dim: int = None,
    vae_hidden_dims=None,
) -> dict:
    """Return the classifier dict EnsembleDynamics expects, for any estimator type.

    target_dim: forwarded to neuralode only, for checkpoints whose metadata lacks
    it and whose path doesn't contain a recognized D4RL env name (e.g. abiomed).
    vae_hidden_dims: forwarded to vae only, for checkpoints whose metadata lacks
    'hidden_dims' and whose architecture doesn't match VAE.load_model's [256, 128]
    default (e.g. abiomed_vae, which is [256, 256]).
    """
    kind = guardian_type or infer_guardian_type(classifier_path)
    print(f"[guardian] type={kind} path={classifier_path}")

    if kind == "realnvp":
        # This repo's vendored copy -- no GORMPO checkout, no extra deps.
        from realnvp_module.realnvp import RealNVP

        info = RealNVP.load_model(classifier_path, device=device)

    elif kind == "vae":
        _use_gormpo(gormpo_root)
        from vae_module.vae import VAE

        vae_kwargs = {"device": device}
        if vae_hidden_dims is not None:
            vae_kwargs["hidden_dims"] = vae_hidden_dims
        info = VAE.load_model(classifier_path, **vae_kwargs)

    elif kind == "kde":
        _use_gormpo(gormpo_root)
        from kde_module.kde import PercentileThresholdKDE

        info = PercentileThresholdKDE.load_model(classifier_path, devid=devid)

    elif kind == "neuralode":
        _use_gormpo(gormpo_root)
        from neuralODE.neural_ode_ood import NeuralODEOOD

        info = NeuralODEOOD.load_model(
            save_path=classifier_path.replace("_model.pt", ""),
            device=device,
            target_dim=target_dim,
        )
        # NeuralODEOOD calls it 'threshold'; EnsembleDynamics reads 'thr'.
        info["thr"] = info["threshold"]
        # The dopri5 solve allocates with the batch: ~2.1GiB at n=1000, OOM by n=10000 on a
        # free 24GiB card. COMBO hands it rollout_batch_size=50000, so it must be chunked.
        info["model"] = _ChunkedScorer(info["model"], chunk_size)

    elif kind == "diffusion":
        root = _use_gormpo(gormpo_root)
        from diffusion.monte_carlo_sampling_unconditional import build_model_from_ckpt
        from diffusers.schedulers.scheduling_ddim import DDIMScheduler
        from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

        ckpt_path = classifier_path
        if os.path.isdir(ckpt_path):  # guardians ship as a dir; the ckpt lives inside
            for name in ("checkpoint.pt", "model.pt"):
                if os.path.exists(os.path.join(ckpt_path, name)):
                    ckpt_path = os.path.join(ckpt_path, name)
                    break
            else:
                raise FileNotFoundError(f"No checkpoint.pt or model.pt under {classifier_path}")

        model, _cfg = build_model_from_ckpt(ckpt_path, device)
        sched_dir = os.path.join(os.path.dirname(ckpt_path), "scheduler")
        try:
            scheduler = DDIMScheduler.from_pretrained(sched_dir)
        except Exception:
            try:
                scheduler = DDPMScheduler.from_pretrained(sched_dir)
            except Exception as exc:
                print(f"[guardian] could not load scheduler ({exc}); using DDIM defaults")
                scheduler = DDIMScheduler(
                    num_train_timesteps=1000,
                    beta_schedule="linear",
                    prediction_type="epsilon",
                )

        import torch

        ckpt = torch.load(ckpt_path, map_location=device)
        target_dim = ckpt.get("target_dim")
        wrapper = _DiffusionDensityWrapper(model, scheduler, target_dim, device, root)

        # Threshold: some checkpoints (e.g. abiomed) embed it directly from training.
        # The sparse D4RL ones don't -- GORMPO reads those from the task's ELBO metrics
        # file instead, relative to its repo root.
        thr = ckpt.get("threshold")
        candidates = ckpt.get("threshold_candidates") or {}
        if thr is None:
            thr = 0.0
            if task:
                prefix = task.lower().split("_")[0].split("-")[0]
                thr_path = os.path.join(
                    root, "diffusion/monte_carlo_results", f"{prefix}_unconditional_ddpm/elbo_metrics.json"
                )
                if os.path.exists(thr_path):
                    with open(thr_path, "r", encoding="utf-8") as f:
                        thr = json.load(f).get("percentile_1.0_logp", 0.0)
                else:
                    print(f"[guardian] no elbo_metrics.json at {thr_path}; thr=0.0")

            sidecar = os.path.join(os.path.dirname(ckpt_path), "checkpoint_metadata.pkl")
            if os.path.exists(sidecar):
                with open(sidecar, "rb") as f:
                    candidates = pickle.load(f).get("threshold_candidates", {}) or {}

        info = {"model": wrapper, "thr": thr, "threshold_candidates": candidates}

    else:
        raise ValueError(f"Unknown guardian type '{kind}'; expected one of {GUARDIAN_TYPES}")

    if threshold_percentile is not None:
        candidates = info.get("threshold_candidates", {}) or {}
        if threshold_percentile in candidates:
            info["thr"] = candidates[threshold_percentile]
            print(f"[guardian] thr <- threshold_candidates[{threshold_percentile}] = {info['thr']:.4f}")
        else:
            print(
                f"[guardian] percentile {threshold_percentile} not in {list(candidates.keys())}; "
                "keeping default thr"
            )

    # Deliberately do NOT set 'name'. In EnsembleDynamics._return_kde_penalty it is not an
    # on/off flag -- it selects the feature convention:
    #     name is None -> input = [state, action], i.e. [next_obs, action]  (these guardians)
    #     name set     -> input = [action, reward]                          (Abiomed guardians)
    # Every guardian under */_medium_expert_sparse_3/ is trained on [next_obs, action], so
    # 'name' must stay unset or the penalty is computed in the wrong space.
    if info.get("name") is not None:
        print(f"[guardian] WARNING: loader set name={info['name']!r}; that selects the "
              "[action, reward] Abiomed feature convention, not [next_obs, action].")
    print(f"[guardian] thr={info['thr']} mean={info.get('mean')} std={info.get('std')} "
          f"name={info.get('name')} (None => [next_obs, action] features)")
    return info


class _DiffusionDensityWrapper:
    """score_samples interface over a diffusion model's ELBO (GORMPO/train.py:45)."""

    def __init__(self, model, scheduler, target_dim, device, gormpo_root):
        self.model = model
        self.scheduler = scheduler
        self.target_dim = target_dim
        self.device = device
        self._gormpo_root = gormpo_root

    def score_samples(self, x, device=None):
        import numpy as np
        import torch

        from diffusion.ddim_training_unconditional import log_prob_elbo

        device = device or self.device
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        with torch.no_grad():
            return log_prob_elbo(
                model=self.model,
                scheduler=self.scheduler,
                x0=x.to(device),
                num_inference_steps=100,
                device=device,
            )


class _ChunkedScorer:
    """score_samples in fixed-size chunks.

    GORMPO's configs set rollout_batch_size=100, so its guardians were only ever asked
    for ~100 samples at a time. COMBO inherits OfflineRL-Kit's rollout_batch_size=50000,
    500x larger -- enough to OOM a 24GiB card on the neuralODE solve, whose memory scales
    with batch. Chunking keeps the results identical and the footprint bounded.
    """

    def __init__(self, inner, chunk_size: int):
        self._inner = inner
        self._chunk_size = int(chunk_size)

    def score_samples(self, x, device=None):
        import numpy as np
        import torch

        n = len(x)
        if n <= self._chunk_size:
            return self._inner.score_samples(x, device)
        out = []
        for i in range(0, n, self._chunk_size):
            part = self._inner.score_samples(x[i : i + self._chunk_size], device)
            out.append(part.detach().cpu() if isinstance(part, torch.Tensor) else np.asarray(part))
        return torch.cat(out) if isinstance(out[0], torch.Tensor) else np.concatenate(out)

    def __getattr__(self, item):  # stay transparent for anything else the model exposes
        return getattr(self._inner, item)
