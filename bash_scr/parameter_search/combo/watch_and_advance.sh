#!/usr/bin/env bash
# One shot, meant to be invoked repeatedly (e.g. every 10 min): checks the 3-way COMBO
# alpha (cql-weight) sweep from alpha_sweep_mcs.sh, restarts any crashed alpha value on
# the GPU with the most free memory, and -- once all 3 have a 200,000-step policy -- picks
# the winner by HELD-OUT eval (eval_combo_mcs.py, not the noisy in-training eval) and
# launches the 5-guardian DBG phase (dbg_all_guardians_mcs.sh) with that winning alpha.
# Never touches alpha_sweep_mcs.sh itself or interferes with an already-running alpha job
# -- a "crashed" restart only fires when a job is neither running nor done.
#
# Idempotent: once the DBG phase has been launched, further invocations just report that
# and exit 0 -- safe to keep invoking this on a timer without watching for it yourself.
#
#   bash bash_scr/parameter_search/combo/watch_and_advance.sh
set -uo pipefail   # no -e: report and keep going rather than die on the first hiccup

REPO="/home/brian/repos/OfflineRL-Kit2"
cd "$REPO"
ALPHAS="0.5 1.0 5.0"
SEED=42
LOG_DIR="log/parameter_search_combo"
DONE_MARKER="$LOG_DIR/DBG_PHASE_LAUNCHED"
MAX_RETRIES=3
mkdir -p "$LOG_DIR"

if [ -f "$DONE_MARKER" ]; then
    echo "ALREADY_ADVANCED: $(cat "$DONE_MARKER")"
    exit 0
fi

freest_gpu() {
    nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits 2>/dev/null \
        | sort -t, -k2 -n -r | head -1 | cut -d, -f1 | tr -d ' '
}

run_dir_for() {
    ls -dt "log/abiomed/combo_noguard_alpha${1}/seed_${SEED}&timestamp_"* 2>/dev/null | head -1 || true
}

all_done=1
for a in $ALPHAS; do
    d="$(run_dir_for "$a")"
    if [ -n "$d" ] && [ -f "$d/model/policy_200000.pth" ]; then
        echo "alpha=$a: DONE ($d)"
        continue
    fi
    all_done=0
    if pgrep -f "algo-name combo_noguard_alpha${a} " >/dev/null 2>&1; then
        echo "alpha=$a: running ($d)"
        continue
    fi

    ctr_file="$LOG_DIR/.retry_count_alpha_${a}"
    n=$(cat "$ctr_file" 2>/dev/null || echo 0)
    if [ "$n" -ge "$MAX_RETRIES" ]; then
        echo "alpha=$a: NOT running, NOT done, and already retried $n times -- giving up, needs a human look" >&2
        continue
    fi
    gpu="$(freest_gpu)"
    [ -n "$gpu" ] || { echo "alpha=$a: NOT running, NOT done -- but couldn't read nvidia-smi, skipping restart this round" >&2; continue; }
    echo "$((n+1))" > "$ctr_file"
    echo "alpha=$a: NOT running and NOT done -- restart attempt $((n+1))/$MAX_RETRIES on GPU $gpu"
    ALPHAS="$a" GPUS="$gpu" SEED="$SEED" bash bash_scr/parameter_search/combo/alpha_sweep_mcs.sh
done

if [ "$all_done" -ne 1 ]; then
    echo "STATUS: not all alphas done yet"
    exit 0
fi

echo "STATUS: all 3 alphas done -- running held-out eval (eval_combo_mcs.py) to pick the winner"
results="$LOG_DIR/winner_eval_$(date +%m%d-%H%M%S).csv"
echo "alpha,run_dir,n_episodes,return,sem" > "$results"
for a in $ALPHAS; do
    d="$(run_dir_for "$a")"
    gpu="$(freest_gpu)"
    row=$(python run_example/eval_combo_mcs.py --policy-path "$d/model/policy_200000.pth" \
            --eval_episodes 1000 --seed 42 --device "cuda:${gpu:-0}" 2>>"$LOG_DIR/winner_eval.err" \
          | grep '^CSVROW,' || true)
    if [ -z "$row" ]; then
        echo "ERROR: held-out eval failed for alpha=$a -- see $LOG_DIR/winner_eval.err" >&2
        exit 1
    fi
    echo "${a},${d},${row#CSVROW,}" >> "$results"
    echo "  alpha=$a -> ${row#CSVROW,}"
done

best=$(python3 - "$results" <<'PY'
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1])))
best = max(rows, key=lambda r: float(r["return"]))
print(best["alpha"])
PY
)
[ -n "$best" ] || { echo "ERROR: could not determine a winner from $results" >&2; exit 1; }
echo "WINNER: alpha (cql-weight) = $best"
echo "Full comparison: $REPO/$results"

echo ">>> launching 5-guardian DBG phase with cql-weight=$best"
CQL_WEIGHT="$best" SEED="$SEED" bash bash_scr/parameter_search/combo/dbg_all_guardians_mcs.sh
echo "winner=$best  source=$REPO/$results  launched=$(date -Iseconds)" > "$DONE_MARKER"
echo "DBG_PHASE_LAUNCHED: winner alpha=$best"
