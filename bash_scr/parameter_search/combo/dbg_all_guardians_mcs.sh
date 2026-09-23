#!/usr/bin/env bash
# Phase (2) of "tune alpha on COMBO without dbg on MCS, then reuse it in the dbg
# experiments": all 5 guardians (kde, vae, ddpm/diffusion, neuralode, realnvp) trained in
# PARALLEL, one dedicated GPU each. Each guardian itself runs its existing 3-seed
# (42 123 456) SEQUENTIAL sweep via that guardian's own script, unmodified:
#   run_example/bash_scr/mult_seed/combo_dbg_{kde,vae,ddpm,neuralode}_mcs.sh
#   run_example/bash_scr/mult_seed/OLD_combo_dbg_realnvp_mcs.sh
# This is a thin launcher, not a reimplementation -- guardian checkpoint paths, tuned
# penalty-coefs, and existence checks all still live in those 5 scripts. We only override
# DEVICE (so each gets its own GPU) and EXTRA_ARGS (to thread the winning --cql-weight and
# the --epoch cap through to every one of that guardian's 3 seeds).
#
#   CQL_WEIGHT=1.0 bash bash_scr/parameter_search/combo/dbg_all_guardians_mcs.sh
#
# CQL_WEIGHT (required) -- the alpha picked by the phase-1 sweep.
# EPOCH=200 (default -- 200,000 steps, same cap as phase 1, applied to EACH of the 3 seeds).
# SEEDS="42 123 456" (default, each guardian script's own) / GPUS="1 3 4 5 7" to pin specific
# GPUs instead of auto-picking the 5 freest at launch time. DETACH=0 to stay attached.
set -euo pipefail

SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
REPO="/home/brian/repos/OfflineRL-Kit2"
cd "$REPO"

CQL_WEIGHT="${CQL_WEIGHT:?set CQL_WEIGHT to the phase-1 winner, e.g. CQL_WEIGHT=1.0}"
EPOCH="${EPOCH:-200}"
SEEDS="${SEEDS:-42 123 456}"
LOG_DIR="log/parameter_search_combo"
mkdir -p "$LOG_DIR"

# Detach so the whole 5-way parallel run survives the terminal/SSH session closing --
# same pattern as alpha_sweep_mcs.sh and combo_dbg_all_estimators_mcs.sh.
if [ "${DETACH:-1}" = 1 ] && [ -z "${DBG5_DETACHED:-}" ]; then
    LOGFILE="$REPO/$LOG_DIR/dbg5_$(date +%m%d-%H%M%S).log"
    DBG5_DETACHED=1 setsid nohup bash "$SELF" "$@" >"$LOGFILE" 2>&1 </dev/null &
    echo "detached: pid $!"
    echo "  tail -f $LOGFILE"
    exit 0
fi

# 5 distinct GPUs, freest-memory-first, unless pinned via GPUS.
if [ -n "${GPUS:-}" ]; then
    read -r -a gpus <<< "$GPUS"
else
    mapfile -t gpus < <(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
        | sort -t, -k2 -n -r | head -5 | cut -d, -f1 | tr -d ' ')
fi
[ "${#gpus[@]}" -eq 5 ] || { echo "ERROR: need 5 GPUs, got ${#gpus[@]}: ${gpus[*]:-none}" >&2; exit 1; }

# heaviest guardian (own ODE-solve guardian net) first, onto the freest GPU.
guardians=(neuralode ddpm vae kde realnvp)
script_for() {
    case "$1" in
        realnvp) echo "run_example/bash_scr/mult_seed/OLD_combo_dbg_realnvp_mcs.sh" ;;
        *)       echo "run_example/bash_scr/mult_seed/combo_dbg_$1_mcs.sh" ;;
    esac
}

echo "============================================"
echo "COMBO+DBG, all 5 guardians in parallel, 3 seeds ($SEEDS) sequential each"
echo "cql-weight: $CQL_WEIGHT  |  epoch: $EPOCH (200,000 steps/seed)"
for i in "${!guardians[@]}"; do echo "  ${guardians[$i]} -> cuda:${gpus[$i]}"; done
echo "============================================"

pids=()
for i in "${!guardians[@]}"; do
    g="${guardians[$i]}"; gpu="${gpus[$i]}"
    s="$(script_for "$g")"
    [ -f "$s" ] || { echo "ERROR: missing script for guardian $g: $s" >&2; exit 1; }
    logf="$REPO/$LOG_DIR/dbg_${g}_$(date +%m%d-%H%M%S).log"
    echo ">>> launching $g on cuda:$gpu -> $logf"
    ( DEVICE="cuda:$gpu" SEEDS="$SEEDS" EXTRA_ARGS="--cql-weight $CQL_WEIGHT --epoch $EPOCH ${EXTRA_ARGS:-}" \
      bash "$s" ) > "$logf" 2>&1 &
    pids+=($!)
    sleep 5   # gentle stagger on shared setup I/O, same reasoning as the other sweeps
done

fail=0
for i in "${!guardians[@]}"; do
    wait "${pids[$i]}" || { echo "ERROR: guardian ${guardians[$i]} exited non-zero" >&2; fail=1; }
done

echo "All 5 guardians finished (3 seeds x 200,000 steps each)."
echo "Per-job logs:       $REPO/$LOG_DIR/dbg_*_*.log"
echo "Per-guardian summaries (held-out eval, mean+/-std across seeds):"
echo "                    $REPO/log/combo_dbg_mcs_mult_seed/*_multiseed_*.csv"
exit "$fail"
