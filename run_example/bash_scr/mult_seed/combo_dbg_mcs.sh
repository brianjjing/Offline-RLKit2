#!/bin/bash
# Multi-seed COMBO + density guardian (DBG) on the Abiomed MCS digital twin.
# Runs run_combo_dbg_mcs.py over 3 seeds; logs per-seed and across-seed return/std.
# Uses the pre-trained RealNVP guardian for abiomed (nothing to train here).
set -e

# use the COMBO conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate COMBO

REPO="/home/brian/repos/OfflineRL-Kit2"
SCRIPT="run_example/run_combo_dbg_mcs.py"
seeds=(42 123 456)          # GORMPO's mult_seed convention

# --- DBG guardian config ---
# base path; RealNVP.load_model adds _model.pth / _meta_data.pkl (input_dim=73 = obs72+act1)
GUARDIAN="/public/gormpo/models/abiomed/realnvp/abiomed_realnvp"
PENALTY_COEF=0.2            # GORMPO abiomed RealNVP config (cormpo/config/real/mbpo_realnvp.yaml)
PENALTY_TYPE=tanh           # GORMPO abiomed RealNVP config

# offlinerlkit's ROOT_DIR is the literal "log" relative to CWD, so run from the repo
# root -> logs land in $REPO/log. PYTHONPATH lets `import offlinerlkit` resolve.
cd "$REPO"
export PYTHONPATH="$REPO"

# preflight: fail loudly now, not 30s into training
[ -f "${GUARDIAN}_model.pth" ] || { echo "ERROR: guardian not found: ${GUARDIAN}_model.pth" >&2; exit 1; }

timestamp=$(date +"%m%d_%H%M%S")
results_dir="log/combo_dbg_mcs_mult_seed"
mkdir -p "$results_dir"
summary="${results_dir}/multiseed_${timestamp}.csv"
echo "seed,return,return_std" > "$summary"

echo "============================================"
echo "Multi-Seed COMBO+DBG Training: Abiomed MCS"
echo "seeds: ${seeds[*]}  |  penalty=${PENALTY_TYPE}x${PENALTY_COEF}"
echo "summary: $REPO/$summary"
echo "============================================"

for seed in "${seeds[@]}"; do
    echo ">>> Training COMBO+DBG (seed=$seed)"
    python "$SCRIPT" \
        --classifier-path "$GUARDIAN" \
        --penalty-coef "$PENALTY_COEF" \
        --penalty-type "$PENALTY_TYPE" \
        --seed "$seed"

    # newest log dir for this seed (make_log_dirs stamps  seed_<seed>&timestamp_<ts>)
    csv=$(ls -dt "log/abiomed/combo/seed_${seed}&timestamp_"*"/record/policy_training_progress.csv" 2>/dev/null | head -1)
    [ -f "$csv" ] || { echo "ERROR: no results CSV for seed $seed (did training finish?)" >&2; exit 1; }

    # per-seed return = mean of last 10 eval epochs == COMBO's own 'last_10_performance' headline
    python - "$csv" "$seed" >> "$summary" <<'PY'
import csv, sys, statistics as st
rows = list(csv.DictReader(open(sys.argv[1])))
last = rows[-10:]
ret = st.mean(float(r["eval/normalized_episode_reward"])     for r in last)
std = st.mean(float(r["eval/normalized_episode_reward_std"]) for r in last)
print(f"{sys.argv[2]},{ret:.4f},{std:.4f}")
PY
    echo "OK seed $seed -> $(tail -1 "$summary")"
done

# headline: mean +/- std of the per-seed returns (the across-seed number you report)
python - "$summary" <<'PY'
import csv, sys, statistics as st
rets = [float(r["return"]) for r in csv.DictReader(open(sys.argv[1]))]
mean = st.mean(rets)
std  = st.stdev(rets) if len(rets) > 1 else 0.0
open(sys.argv[1], "a").write(f"mean_over_seeds,{mean:.4f},{std:.4f}\n")
print(f"\n=== {len(rets)} seeds: return = {mean:.4f} +/- {std:.4f} ===")
PY

echo "Summary written to $REPO/$summary"
