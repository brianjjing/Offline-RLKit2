#!/bin/bash
# Multi-seed COMBO+DBG (KDE density guardian) on the Abiomed MCS digital twin.
# Sibling of combo_dbg_vae_mcs.sh / combo_dbg_ddpm_mcs.sh / combo_dbg_neuralode_mcs.sh /
# OLD_combo_dbg_realnvp_mcs.sh -- same structure, different guardian. Split out of
# combo_dbg_all_estimators_mcs.sh (which runs all 4 in parallel, but seed 42 only).
#
#   SEEDS="44 43 42" bash run_example/bash_scr/mult_seed/combo_dbg_kde_mcs.sh
#
# Per-seed headline comes from eval_combo_mcs.py on the HELD-OUT test split, not
# from eval/normalized_episode_reward in the training CSV -- the in-training eval
# calls env.reset() with no idx, so it still scores windows the policy trained on.
#
# Dynamics are NOT reused across seeds here (unlike combo_dbg_all_estimators_mcs.sh's
# reuse across estimators) -- --seed also seeds the dynamics ensemble's own init, so a
# real multi-seed sweep needs each seed to train its own, same as OLD_combo_dbg_realnvp_mcs.sh.
set -e

source ~/miniconda3/etc/profile.d/conda.sh
conda activate COMBO

REPO="/home/brian/repos/OfflineRL-Kit2"
SCRIPT="run_example/run_combo_dbg_mcs.py"
export GORMPO_ROOT="${GORMPO_ROOT:-/home/brian/repos/GORMPO}"
seeds=(${SEEDS:-42 123 456})
DEVICE=${DEVICE:-cuda:7}
EVAL_EPISODES=${EVAL_EPISODES:-1000}
# eval env seed: pinned so every seed is scored on the identical test windows
EVAL_SEED=${EVAL_SEED:-42}
EXTRA_ARGS=${EXTRA_ARGS:-}
GUARDIAN_TYPE=kde
GUARDIAN=${GUARDIAN:-/public/gormpo/models/abiomed/kde/trained_kde_1}
# GORMPO_abiomed's own tuned coefficient for MCS (cormpo/config/real/mbpo_kde.yaml), read
# from LEQ2/bash_scr/leq_dbg/LEQ_DBG_MCS.sh.
PENALTY_COEF=${PENALTY_COEF:-0.2}
PENALTY_TYPE=${PENALTY_TYPE:-tanh}

cd "$REPO"
export PYTHONPATH="$REPO"

timestamp=$(date +"%m%d_%H%M%S")
results_dir="log/combo_dbg_mcs_mult_seed"
mkdir -p "$results_dir"
seedtag=$(IFS=-; echo "${seeds[*]}")
summary="${results_dir}/kde_multiseed_seed${seedtag}_${timestamp}.csv"
echo "seed,n_episodes,return,sem" > "$summary"

echo "============================================"
echo "Multi-Seed COMBO+DBG: Abiomed MCS (KDE)"
echo "seeds: ${seeds[*]}  |  device: $DEVICE  |  extra: ${EXTRA_ARGS:-none}"
echo "summary: $REPO/$summary"
echo "============================================"

[ -f "${GUARDIAN}.faiss" ] && [ -f "${GUARDIAN}_metadata.pkl" ] || {
    echo "ERROR: guardian not found: ${GUARDIAN}.faiss / _metadata.pkl" >&2; exit 1; }

for seed in "${seeds[@]}"; do
    echo ">>> Training COMBO+DBG (seed=$seed)"
    python "$SCRIPT" --seed "$seed" --device "$DEVICE" \
        --classifier-path "$GUARDIAN" --guardian-type "$GUARDIAN_TYPE" \
        --penalty-coef "$PENALTY_COEF" --penalty-type "$PENALTY_TYPE" $EXTRA_ARGS

    # guardian_type is baked into the log dir (run_combo_dbg_mcs.py's make_log_dirs call),
    # so this glob only ever matches KDE runs, further narrowed to this seed.
    d=$(ls -dt "log/abiomed/combo&guardian_type=${GUARDIAN_TYPE}/seed_${seed}&timestamp_"* 2>/dev/null | head -1)
    [ -f "$d/model/policy.pth" ] || { echo "ERROR: no policy for seed $seed" >&2; exit 1; }

    echo ">>> Evaluating seed=$seed on held-out test windows"
    row=$(python run_example/eval_combo_mcs.py \
            --policy-path "$d/model/policy.pth" \
            --eval_episodes "$EVAL_EPISODES" --seed "$EVAL_SEED" --device "$DEVICE" \
          | grep '^CSVROW,')
    [ -n "$row" ] || { echo "ERROR: eval failed for seed $seed" >&2; exit 1; }
    echo "${seed},${row#CSVROW,}" >> "$summary"
    echo "OK seed $seed -> $(tail -1 "$summary")"
done

# headline: mean +/- std of the per-seed held-out returns
python - "$summary" <<'PY'
import csv, sys, statistics as st
rows = list(csv.DictReader(open(sys.argv[1])))
r = [float(x["return"]) for x in rows]
for x in rows:
    print(f'  seed {x["seed"]}: {float(x["return"]):+.4f}  (sem {float(x["sem"]):.4f}, n={x["n_episodes"]})')
print(f'\n=== {len(r)} seeds: return = {st.mean(r):.4f} +/- {st.stdev(r) if len(r)>1 else 0.0:.4f} ===')
PY
echo "Summary written to $REPO/$summary"
