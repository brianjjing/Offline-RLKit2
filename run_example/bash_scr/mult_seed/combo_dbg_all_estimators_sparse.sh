#!/usr/bin/env bash
# COMBO + density guardian (DBG) across all 5 estimators, on the 3 sparse D4RL tasks.
# Trains the checkpoints the 6-panel t-SNE figure needs (notebooks/run_tsne_policy_vs_dataset_all.sh).
#
# Seed 42 only -- that is what the t-SNE discovery defaults to (--seed-filter 42). The existing
# per-task scripts (combo_dbg_{halfcheetah,hopper,walker2d}_sparse.sh) still cover the 3-seed
# realnvp sweep; this one is the estimator sweep.
set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate COMBO

REPO="/home/brian/repos/OfflineRL-Kit2"
SCRIPT="run_example/run_combo_dbg_sparse_d4rl.py"
cd "$REPO"
export PYTHONPATH="$REPO"
# vae/kde/diffusion/neuralode are imported from the GORMPO checkout; realnvp is vendored here.
export GORMPO_ROOT="${GORMPO_ROOT:-/home/brian/repos/GORMPO}"

SEED="${SEED:-42}"
# run_combo_dbg_sparse_d4rl.py defaults to "cuda" == cuda:0, which is usually the busiest card.
# Pin explicitly: DEVICE=cuda:5, or CUDA_VISIBLE_DEVICES=5 with DEVICE=cuda.
DEVICE="${DEVICE:-cuda}"
# Only binds for neuralode. 4096 -> ~58h guardian overhead at 9.4GiB; 8192 -> ~30h at 18.4GiB
# (too tight to share a 24GiB card with COMBO itself).
GUARDIAN_CHUNK_SIZE="${GUARDIAN_CHUNK_SIZE:-4096}"
# Estimators to sweep. realnvp is already trained for all 3 tasks -- drop it from this list
# unless you want to reproduce it. Order matches the t-SNE panel order.
ESTIMATORS="${ESTIMATORS:-kde vae diffusion neuralode}"

# --- per-task config, lifted verbatim from the existing combo_dbg_<task>_sparse.sh scripts ---
# task | dataset | guardian dir | rollout-length | cql-weight | real-ratio
TASKS=(
  "halfcheetah-medium-expert-v2|/public/d4rl/sparse_datasets/halfcheetah_medium_expert_sparse_72.5.pkl|/public/gormpo/models/halfcheetah_medium_expert_sparse_3|5|5.0|0.5"
  "hopper-medium-expert-v2|/public/d4rl/sparse_datasets/hopper_medium_expert_sparse_78.pkl|/public/gormpo/models/hopper_medium_expert_sparse_3|5|5.0|0.5"
  "walker2d-medium-expert-v2|/public/d4rl/sparse_datasets/walker2d_medium_expert_sparse_73.pkl|/public/gormpo/models/walker2d_medium_expert_sparse_3|1|5.0|0.5"
)
PENALTY_TYPE="${PENALTY_TYPE:-tanh}"

# Penalty coefficient is tuned PER ESTIMATOR, not per task -- the estimators' log-prob
# scales differ by orders of magnitude (on the same input: kde ~-23, vae ~-48,
# realnvp ~-1e4, diffusion ~-3e5), so one shared coef is not comparable across them.
# Values are GORMPO's tuned settings from configs/<est>/gormpo_<task>_medium_expert_sparse_3.yaml
# (each the best of its 0.1/0.3/0.5/0.7 sweep). The realnvp column reproduces the 0.8/0.8/0.5
# already used by combo_dbg_<task>_sparse.sh, so existing runs stay consistent.
#                        realnvp  kde    vae    diffusion  neuralode
# halfcheetah              0.8    0.1    0.1      0.3         0.1
# hopper                   0.8    0.05   0.3      0.05        0.5
# walker2d                 0.5    0.5    0.5      0.05        0.05
penalty_coef () {
  local task="$1" est="$2"
  case "${task%%-*}:${est}" in
    halfcheetah:realnvp) echo 0.8 ;;  halfcheetah:kde)   echo 0.1  ;; halfcheetah:vae)       echo 0.1  ;;
    halfcheetah:diffusion) echo 0.3 ;; halfcheetah:neuralode) echo 0.1 ;;
    hopper:realnvp)      echo 0.8 ;;  hopper:kde)        echo 0.05 ;; hopper:vae)            echo 0.3  ;;
    hopper:diffusion)    echo 0.05 ;; hopper:neuralode)  echo 0.5  ;;
    walker2d:realnvp)    echo 0.5 ;;  walker2d:kde)      echo 0.5  ;; walker2d:vae)          echo 0.5  ;;
    walker2d:diffusion)  echo 0.05 ;; walker2d:neuralode) echo 0.05 ;;
    *) echo "ERROR: no tuned penalty-coef for ${task}/${est}" >&2; return 1 ;;
  esac
}

# Guardian checkpoints are named after the estimator, except neuralode's dir is camelCase.
guardian_path () {
  local dir="$1" est="$2"
  case "$est" in
    neuralode) echo "${dir}/neuralODE" ;;
    *)         echo "${dir}/${est}" ;;
  esac
}

# preflight: fail loudly now, not 30s into a multi-hour run
for row in "${TASKS[@]}"; do
  IFS='|' read -r task dataset gdir _ _ _ <<< "$row"
  [ -f "$dataset" ] || { echo "ERROR: dataset not found: $dataset" >&2; exit 1; }
  for est in $ESTIMATORS; do
    g="$(guardian_path "$gdir" "$est")"
    # realnvp/vae ship as <base>_model.pth; kde as <base>.faiss; diffusion/neuralode as a dir.
    if [ ! -e "${g}_model.pth" ] && [ ! -e "${g}.faiss" ] && [ ! -d "$g" ]; then
      echo "ERROR: guardian not found for $task/$est at $g" >&2; exit 1
    fi
    penalty_coef "$task" "$est" >/dev/null || exit 1
  done
done
echo "preflight OK (device=${DEVICE}): $(echo $ESTIMATORS | wc -w) estimators x ${#TASKS[@]} tasks, seed ${SEED}"

for row in "${TASKS[@]}"; do
  IFS='|' read -r task dataset gdir rollout cql real_ratio <<< "$row"
  for est in $ESTIMATORS; do
    g="$(guardian_path "$gdir" "$est")"
    penalty="$(penalty_coef "$task" "$est")"
    echo ">>> COMBO+DBG $task est=$est (seed=$SEED, penalty=${PENALTY_TYPE}x${penalty})"
    python "$SCRIPT" \
      --task "$task" \
      --dataset-path "$dataset" \
      --classifier-path "$g" \
      --guardian-type "$est" \
      --penalty-coef "$penalty" \
      --penalty-type "$PENALTY_TYPE" \
      --rollout-length "$rollout" \
      --cql-weight "$cql" \
      --real-ratio "$real_ratio" \
      --device "$DEVICE" \
      --guardian-chunk-size "$GUARDIAN_CHUNK_SIZE" \
      --seed "$SEED"
  done
done

echo "All runs completed. Now: bash notebooks/run_tsne_policy_vs_dataset_all.sh"
