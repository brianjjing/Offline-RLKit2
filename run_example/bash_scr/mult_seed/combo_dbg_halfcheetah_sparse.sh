#!/bin/bash
# COMBO + density guardian (DBG) on the sparse halfcheetah-medium-expert dataset, 3 seeds.
# Uses the pre-trained RealNVP guardian (nothing to train here).
set -e

source ~/miniconda3/etc/profile.d/conda.sh
conda activate COMBO

REPO="/home/brian/repos/OfflineRL-Kit2"
SCRIPT="run_example/run_combo_dbg_sparse_d4rl.py"
cd "$REPO"
export PYTHONPATH="$REPO"

# --- per-env config (dataset <-> guardian pairing is GORMPO's gormpo_*_sparse_3 config) ---
TASK="halfcheetah-medium-expert-v2"
DATASET="/public/d4rl/sparse_datasets/halfcheetah_medium_expert_sparse_72.5.pkl"
GUARDIAN="/public/gormpo/models/halfcheetah_medium_expert_sparse_3/realnvp"  # base path; load_model adds _model.pth/_meta_data.pkl
ROLLOUT_LENGTH=5          # COMBO medium-expert convention
CQL_WEIGHT=5.0
PENALTY_COEF=0.8          # GORMPO-tuned guardian scale; consider lowering for COMBO (CQL already regularizes)
PENALTY_TYPE=tanh
seeds=(42 123 456)

# preflight: fail loudly now, not 30s into training
[ -f "$DATASET" ]              || { echo "ERROR: dataset not found: $DATASET" >&2; exit 1; }
[ -f "${GUARDIAN}_model.pth" ] || { echo "ERROR: guardian not found: ${GUARDIAN}_model.pth" >&2; exit 1; }

for seed in "${seeds[@]}"; do
    echo ">>> COMBO+DBG $TASK (seed=$seed, penalty=${PENALTY_TYPE}x${PENALTY_COEF})"
    python "$SCRIPT" \
        --task "$TASK" \
        --dataset-path "$DATASET" \
        --classifier-path "$GUARDIAN" \
        --penalty-coef "$PENALTY_COEF" \
        --penalty-type "$PENALTY_TYPE" \
        --rollout-length "$ROLLOUT_LENGTH" \
        --cql-weight "$CQL_WEIGHT" \
        --seed "$seed"
done
