#!/usr/bin/env bash
# COMBO port of GORMPO/notebooks/run_tsne_policy_vs_dataset_all.sh.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# The COMBO conda env has no sklearn/scipy; the GORMPO one has both plus d4rl,
# and offlinerlkit imports fine there via PYTHONPATH. Override with PYTHON_BIN.
PYTHON_BIN="${PYTHON_BIN:-${HOME}/miniconda3/envs/GORMPO/bin/python}"
export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

TASK_HALF="${TASK_HALF:-halfcheetah-medium-expert-v2}"
DATA_HALF="${DATA_HALF:-/public/d4rl/sparse_datasets/halfcheetah_medium_expert_sparse_72.5.pkl}"

TASK_HOP="${TASK_HOP:-hopper-medium-expert-v2}"
DATA_HOP="${DATA_HOP:-/public/d4rl/sparse_datasets/hopper_medium_expert_sparse_78.pkl}"

TASK_WALK="${TASK_WALK:-walker2d-medium-expert-v2}"
DATA_WALK="${DATA_WALK:-/public/d4rl/sparse_datasets/walker2d_medium_expert_sparse_73.pkl}"

# Six density-estimator panels: COMBO, -KDE, -VAE, -RealNVP, -DDPM, -NeuralODE.
PANEL_COLS="${PANEL_COLS:-6}"
LOG_ROOT="${LOG_ROOT:-${ROOT_DIR}/log}"
SEED_FILTER="${SEED_FILTER:-42}"

# Equate support sizes: use the same cap for offline and rollout.
# Combined t-SNE points ~= 2 * MAX_SAMPLES.
MAX_SAMPLES="${MAX_SAMPLES:-100000}"

# Halfcheetah typically has longer episodes; Hopper/Walker often terminate earlier.
N_ROLLOUT_HALF="${N_ROLLOUT_HALF:-100}"
N_ROLLOUT_HOP="${N_ROLLOUT_HOP:-200}"
N_ROLLOUT_WALK="${N_ROLLOUT_WALK:-200}"

SUPPORT_SOURCE="${SUPPORT_SOURCE:-rollout}" # rollout is what you want for policy support overlap

SAVE_PDF_FLAG="--save-pdf"
if [[ "${SAVE_PDF:-1}" -eq 0 ]]; then
  SAVE_PDF_FLAG=""
fi

DETERMINISTIC_FLAG="--deterministic"
if [[ "${DETERMINISTIC:-1}" -eq 0 ]]; then
  DETERMINISTIC_FLAG=""
fi

OUT_DIR="${OUT_DIR:-${ROOT_DIR}/results/tsne_policy_vs_dataset_sparse}"

# Use GPU if available, unless overridden by DEVICE=cpu|cuda.
DEFAULT_DEVICE="cpu"
if command -v nvidia-smi >/dev/null 2>&1; then
  if nvidia-smi -L >/dev/null 2>&1; then
    DEFAULT_DEVICE="cuda"
  fi
fi
DEVICE="${DEVICE:-${DEFAULT_DEVICE}}"

echo "Running t-SNE overlap plots (COMBO policy support vs dataset support)"
echo "Python: ${PYTHON_BIN}"
echo "Logs:   ${LOG_ROOT} (seed ${SEED_FILTER})"
echo "Output: ${OUT_DIR}"
echo "Device: ${DEVICE}"

run_one () {
  local task="$1"
  local dataset_path="$2"
  local n_rollout_episodes="$3"
  echo "---- $task ----"
  "${PYTHON_BIN}" "${ROOT_DIR}/notebooks/tsne_policy_vs_dataset.py" \
    --task "${task}" \
    --dataset-path "${dataset_path}" \
    --log-root "${LOG_ROOT}" \
    --seed-filter "${SEED_FILTER}" \
    --support-source "${SUPPORT_SOURCE}" \
    ${DETERMINISTIC_FLAG} \
    --device "${DEVICE}" \
    --n-rollout-episodes "${n_rollout_episodes}" \
    --max-offline-samples "${MAX_SAMPLES}" \
    --max-rollout-samples "${MAX_SAMPLES}" \
    --panel-cols "${PANEL_COLS}" \
    --output-dir "${OUT_DIR}" \
    ${SAVE_PDF_FLAG}
}

run_one "${TASK_HALF}" "${DATA_HALF}" "${N_ROLLOUT_HALF}"
run_one "${TASK_HOP}"  "${DATA_HOP}"  "${N_ROLLOUT_HOP}"
run_one "${TASK_WALK}" "${DATA_WALK}" "${N_ROLLOUT_WALK}"

echo "All runs completed."
