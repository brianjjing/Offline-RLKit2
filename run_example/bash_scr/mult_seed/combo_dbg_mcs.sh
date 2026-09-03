#!/bin/bash
# Multi-seed COMBO+DBG (RealNVP density guardian) on the Abiomed MCS digital twin.
# run_combo_dbg_mcs.py REQUIRES --classifier-path, so it is passed explicitly below.
#
#   SEEDS="44 43 42" bash run_example/bash_scr/mult_seed/combo_dbg_mcs.sh
#
# Per-seed headline comes from eval_combo_mcs.py on the HELD-OUT test split, not
# from eval/normalized_episode_reward in the training CSV -- the in-training eval
# calls env.reset() with no idx, so it still scores windows the policy trained on.
set -e

source ~/miniconda3/etc/profile.d/conda.sh
conda activate COMBO

REPO="/home/brian/repos/OfflineRL-Kit2"
SCRIPT="run_example/run_combo_dbg_mcs.py"
seeds=(${SEEDS:-42 123 456})
DEVICE=${DEVICE:-cuda:6}
EVAL_EPISODES=${EVAL_EPISODES:-1000}
# eval env seed: pinned so every seed is scored on the identical test windows
EVAL_SEED=${EVAL_SEED:-42}
EXTRA_ARGS=${EXTRA_ARGS:-}
# GORMPO abiomed RealNVP config (cormpo/config/real/mbpo_realnvp.yaml)
GUARDIAN=${GUARDIAN:-/public/gormpo/models/abiomed/realnvp/abiomed_realnvp}
PENALTY_COEF=${PENALTY_COEF:-0.2}
PENALTY_TYPE=${PENALTY_TYPE:-tanh}

cd "$REPO"
export PYTHONPATH="$REPO"

timestamp=$(date +"%m%d_%H%M%S")
results_dir="log/combo_dbg_mcs_mult_seed"
mkdir -p "$results_dir"
seedtag=$(IFS=-; echo "${seeds[*]}")
summary="${results_dir}/multiseed_seed${seedtag}_${timestamp}.csv"
echo "seed,n_episodes,return,sem" > "$summary"

echo "============================================"
echo "Multi-Seed COMBO+DBG: Abiomed MCS"
echo "seeds: ${seeds[*]}  |  device: $DEVICE  |  extra: ${EXTRA_ARGS:-none}"
echo "summary: $REPO/$summary"
echo "============================================"

[ -f "${GUARDIAN}_model.pth" ] || { echo "ERROR: guardian not found: ${GUARDIAN}_model.pth" >&2; exit 1; }

for seed in "${seeds[@]}"; do
    echo ">>> Training COMBO+DBG (seed=$seed)"
    python "$SCRIPT" --seed "$seed" --device "$DEVICE" \
        --classifier-path "$GUARDIAN" --penalty-coef "$PENALTY_COEF" --penalty-type "$PENALTY_TYPE" $EXTRA_ARGS

    # newest log dir for this seed (make_log_dirs stamps seed_<seed>&timestamp_<ts>)
    d=$(ls -dt "log/abiomed/combo/seed_${seed}&timestamp_"* 2>/dev/null | head -1)
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
