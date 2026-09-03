#!/bin/bash
# Multi-seed COMBO on the Abiomed MCS digital twin (no density guardian unless
# --penalty-coef is left at its 0.2 default -- pass PENALTY_COEF=0 to disable).
#
#   SEEDS="44 43 42" bash run_example/bash_scr/mult_seed/combo_mcs.sh
#
# Per-seed headline comes from eval_combo_mcs.py on the HELD-OUT test split, not
# from eval/normalized_episode_reward in the training CSV -- the in-training eval
# calls env.reset() with no idx, so it still scores windows the policy trained on.
set -e

source ~/miniconda3/etc/profile.d/conda.sh
conda activate COMBO

REPO="/home/brian/repos/OfflineRL-Kit2"
SCRIPT="run_example/run_combo_in_mcs.py"
seeds=(${SEEDS:-42 123 456})
DEVICE=${DEVICE:-cuda:6}
EVAL_EPISODES=${EVAL_EPISODES:-1000}
# eval env seed: pinned so every seed is scored on the identical test windows
EVAL_SEED=${EVAL_SEED:-42}
EXTRA_ARGS=${EXTRA_ARGS:-}
# log namespace: make_log_dirs writes log/<task>/<algo-name>/. Give concurrent variants
# distinct names (e.g. ALGO=combo_noguard) so their seed_<n> dirs never interleave.
ALGO=${ALGO:-combo}

cd "$REPO"
export PYTHONPATH="$REPO"

timestamp=$(date +"%m%d_%H%M%S")
results_dir="log/combo_mcs_mult_seed"
mkdir -p "$results_dir"
seedtag=$(IFS=-; echo "${seeds[*]}")
summary="${results_dir}/multiseed_seed${seedtag}_${timestamp}.csv"
echo "seed,n_episodes,return,sem" > "$summary"

echo "============================================"
echo "Multi-Seed COMBO: Abiomed MCS"
echo "seeds: ${seeds[*]}  |  device: $DEVICE  |  algo: $ALGO  |  extra: ${EXTRA_ARGS:-none}"
echo "summary: $REPO/$summary"
echo "============================================"

for seed in "${seeds[@]}"; do
    echo ">>> Training COMBO (seed=$seed)"
    # snapshot before/after instead of `ls -dt | head -1`: dir mtime is set at creation
    # and never updated when files land in model/ or record/, so "newest" picks whichever
    # concurrent run started last, not the one that just finished.
    glob="log/abiomed/${ALGO}/seed_${seed}&timestamp_"
    before=$(ls -d "${glob}"* 2>/dev/null | sort)
    python "$SCRIPT" --seed "$seed" --device "$DEVICE" --algo-name "$ALGO" $EXTRA_ARGS
    d=$(comm -13 <(printf '%s\n' "$before" | grep -v '^$') \
                 <(ls -d "${glob}"* 2>/dev/null | sort) | head -1)

    [ -n "$d" ] && [ -f "$d/model/policy.pth" ] || {
        echo "ERROR: seed $seed produced no policy (looked for a new dir under ${glob}*)" >&2; exit 1; }
    echo "    run dir: $d"

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
