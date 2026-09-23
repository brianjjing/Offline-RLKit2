#!/usr/bin/env bash
# Step (1) of the plan "tune alpha on COMBO without dbg on MCS, then reuse it in the
# dbg experiments": grid search over COMBO's conservatism penalty (the paper calls this
# alpha; this repo's CLI/code calls it --cql-weight, see offlinerlkit/policy/model_based/
# combo.py) across 0.5 / 1.0 / 5.0. No density guardian (--penalty-coef 0 -- verified in
# EnsembleDynamics.step(): penalty_coef==0 skips the guardian entirely, same "no dbg" path
# documented in run_example/bash_scr/mult_seed/combo_mcs.sh). NOT run_combo_in_mcs.py's
# own --alpha flag -- that's SAC's entropy temperature, auto-tuned by default and unrelated
# to conservatism.
#
# Each alpha value trains dynamics+policy for exactly 200 epochs * 1000 steps/epoch =
# 200,000 steps then stops. checkpoint/policy.pth is overwritten every epoch, so a
# background watcher snapshots it once as policy_100000.pth right after epoch 100 (100,000
# steps); the trainer's own end-of-training save gives the 200,000-step model, copied here
# to model/policy_200000.pth for a matching name. Both points' eval/normalized_episode_reward
# (the in-training eval -- see combo_mcs.sh's caveat: not a held-out score) land in the sweep
# summary CSV alongside the full per-epoch curve already in each run's
# record/policy_training_progress.csv.
#
#   bash bash_scr/parameter_search/combo/alpha_sweep_mcs.sh
#
# ALPHAS="0.5 1.0" GPUS="3 4" to override the grid / GPU assignment. DETACH=0 to stay
# in the foreground instead of backgrounding+returning immediately.
set -euo pipefail

SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate COMBO

REPO="/home/brian/repos/OfflineRL-Kit2"
SCRIPT="run_example/run_combo_in_mcs.py"
cd "$REPO"
export PYTHONPATH="$REPO"

ALPHAS="${ALPHAS:-0.5 1.0 5.0}"
SEED="${SEED:-42}"
# one GPU per alpha value, positionally matched. Picked from `nvidia-smi` headroom on
# 2026-09-22 (all 8 GPUs were in shared use; 3/4/5 had the most free memory) -- recheck if
# rerunning much later, other tenants' usage on this shared box fluctuates.
GPUS="${GPUS:-3 4 5}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
LOG_DIR="log/parameter_search_combo"
mkdir -p "$LOG_DIR"

read -r -a alphas <<< "$ALPHAS"
read -r -a gpus <<< "$GPUS"
[ "${#alphas[@]}" -eq "${#gpus[@]}" ] || {
    echo "ERROR: ALPHAS (${#alphas[@]}) and GPUS (${#gpus[@]}) must have the same length" >&2; exit 1; }

# Detach from the terminal/SSH session so the sweep survives it closing -- same pattern as
# run_example/bash_scr/mult_seed/combo_dbg_all_estimators_mcs.sh. DETACH=0 to stay attached.
if [ "${DETACH:-1}" = 1 ] && [ -z "${SWEEP_DETACHED:-}" ]; then
    LOGFILE="$REPO/$LOG_DIR/sweep_$(date +%m%d-%H%M%S).log"
    SWEEP_DETACHED=1 setsid nohup bash "$SELF" "$@" >"$LOGFILE" 2>&1 </dev/null &
    echo "detached: pid $!"
    echo "  tail -f $LOGFILE"
    exit 0
fi

SUMMARY="$LOG_DIR/summary_$(date +%m%d-%H%M%S).csv"
echo "alpha,run_dir,eval_normalized_reward_100000,eval_normalized_reward_200000" > "$SUMMARY"

echo "============================================"
echo "COMBO alpha (cql-weight) grid search, no guardian, MCS"
echo "alphas: ${alphas[*]}  |  gpus: ${gpus[*]}  |  seed: $SEED  |  extra: ${EXTRA_ARGS:-none}"
echo "summary: $REPO/$SUMMARY"
echo "============================================"

run_one() {
    local aw="$1" gpu="$2"
    local algo="combo_noguard_alpha${aw}"
    local logf="$REPO/$LOG_DIR/train_alpha${aw}_$(date +%m%d-%H%M%S).log"

    (
        set -euo pipefail
        echo ">>> COMBO alpha=$aw (--cql-weight)  gpu=$gpu  algo=$algo  seed=$SEED"

        glob="log/abiomed/${algo}/seed_${SEED}&timestamp_"
        # `|| true`: under pipefail, `ls` finding zero matches (the norm for a brand-new
        # algo-name) exits nonzero and would otherwise kill this subshell via errexit before
        # python ever launches -- same guard combo_dbg_all_estimators_mcs.sh uses for this.
        before=$(ls -d "${glob}"* 2>/dev/null | sort || true)

        CUDA_VISIBLE_DEVICES="$gpu" python "$SCRIPT" \
            --cql-weight "$aw" --penalty-coef 0 \
            --algo-name "$algo" --seed "$SEED" --device cuda \
            --epoch 200 $EXTRA_ARGS &
        train_pid=$!

        # find the run dir the training process just created (usually seconds; dataset
        # load can take longer), bailing out early if training has already died.
        d=""
        while kill -0 "$train_pid" 2>/dev/null; do
            d=$(comm -13 <(printf '%s\n' "$before" | grep -v '^$') \
                         <(ls -d "${glob}"* 2>/dev/null | sort) | head -1)
            [ -n "$d" ] && break
            sleep 5
        done

        # background watcher: checkpoint/policy.pth is overwritten every epoch, so grab it
        # once epoch 100 (100,000 steps) has saved, before epoch 101 overwrites it. Stops
        # polling on its own once training exits, successful or not.
        (
            [ -n "$d" ] || { echo "WARN: alpha=$aw never saw a run dir; no 100000-step snapshot" >&2; exit 1; }
            ckpt="$d/checkpoint/policy.pth"
            n=0 last=""
            while [ "$n" -lt 100 ] && kill -0 "$train_pid" 2>/dev/null; do
                if [ -f "$ckpt" ]; then
                    cur=$(stat -c %Y "$ckpt" 2>/dev/null || echo "")
                    if [ -n "$cur" ] && [ "$cur" != "$last" ]; then
                        n=$((n+1)); last="$cur"
                    fi
                fi
                sleep 10
            done
            if [ "$n" -ge 100 ]; then
                cp "$ckpt" "$d/checkpoint/policy_100000.pth"
                echo "    100000-step checkpoint -> $d/checkpoint/policy_100000.pth"
            else
                echo "WARN: alpha=$aw training exited before epoch 100 (saw $n checkpoint saves); no 100000-step snapshot" >&2
            fi
        ) &
        watcher_pid=$!

        wait "$train_pid"; train_rc=$?
        wait "$watcher_pid" || true
        [ "$train_rc" -eq 0 ] || { echo "ERROR: training failed for alpha=$aw (exit $train_rc)" >&2; exit 1; }

        [ -n "$d" ] && [ -f "$d/model/policy.pth" ] || {
            echo "ERROR: alpha=$aw produced no final policy (looked under ${glob}*)" >&2; exit 1; }
        cp "$d/model/policy.pth" "$d/model/policy_200000.pth"
        echo "    200000-step checkpoint -> $d/model/policy_200000.pth"

        row=$(python - "$d/record/policy_training_progress.csv" <<'PY'
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1])))
want = {"100000": "", "200000": ""}
for r in rows:
    if r.get("timestep") in want:
        want[r["timestep"]] = r.get("eval/normalized_episode_reward", "")
print(f'{want["100000"]},{want["200000"]}')
PY
)
        echo "${aw},${d},${row}" >> "$SUMMARY"
        echo "OK alpha=$aw -> $(tail -1 "$SUMMARY")"
    ) > "$logf" 2>&1 &
}

for i in "${!alphas[@]}"; do
    run_one "${alphas[$i]}" "${gpus[$i]}"
    sleep 10   # stagger: all jobs load the same real-data pickle at startup
done
wait

echo "All alphas done."
echo "Summary: $REPO/$SUMMARY"
cat "$SUMMARY"
