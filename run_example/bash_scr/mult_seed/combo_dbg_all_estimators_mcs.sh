#!/usr/bin/env bash
# COMBO + density guardian (DBG) across kde/vae/ddpm(diffusion)/neuralode on the Abiomed
# MCS digital twin. MCS counterpart of combo_dbg_all_estimators_sparse.sh (same estimator
# sweep, sparse D4RL instead); companion to combo_dbg_mcs.sh (realnvp only, 3 policy seeds).
#
# Seed 42 only. Jobs run IN PARALLEL, each pinned to its own GPU -- there's a single task
# here (no per-task loop like the sparse script), so parallelism is free:
#   - neuralode's guardian (dopri5 ODE solve) is the only one with real GPU cost: measured
#     empirically at ~4.3GiB / ~25s per rollout's guardian call at --guardian-chunk-size
#     8192 (vs ~223s at the 4096 default -- chunk size matters much more here than on the
#     sparse D4RL tasks, since abiomed's 73-dim guardian input is 3-5x theirs). It gets its
#     own card.
#   - kde/vae/ddpm are each ~1.7GiB total end-to-end (COMBO's own footprint -- kde's faiss
#     index runs on CPU here, no GPU faiss build in this env; vae/ddpm's guardians are
#     lightweight MLPs). They share a card.
# Both cards had >10GiB free of 24GiB when this was written (2026-09-17); re-check
# nvidia-smi if rerunning much later, other tenants' usage on this shared box fluctuates.
set -euo pipefail

SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate COMBO

REPO="/home/brian/repos/OfflineRL-Kit2"
SCRIPT="run_example/run_combo_dbg_mcs.py"
cd "$REPO"
export PYTHONPATH="$REPO"
export GORMPO_ROOT="${GORMPO_ROOT:-/home/brian/repos/GORMPO}"

SEED="${SEED:-42}"
EVAL_EPISODES="${EVAL_EPISODES:-1000}"
EVAL_SEED="${EVAL_SEED:-42}"
PENALTY_TYPE="${PENALTY_TYPE:-tanh}"
NEURALODE_CHUNK="${NEURALODE_CHUNK:-8192}"
ESTIMATORS="${ESTIMATORS:-kde vae ddpm neuralode}"

# GORMPO_abiomed's own tuned penalty-coef per estimator for MCS (cormpo/config/real/
# mbpo_<estimator>.yaml), read from LEQ2/bash_scr/leq_dbg/LEQ_DBG_MCS.sh -- cross-checked
# against combo_dbg_mcs.sh's realnvp default (0.2), which matches.
penalty_coef() {
  case "$1" in
    kde) echo 0.2 ;; vae) echo 0.1 ;; ddpm) echo 0.4 ;; neuralode) echo 0.2 ;;
    *) echo "ERROR: no tuned penalty-coef for estimator '$1'" >&2; return 1 ;;
  esac
}

# Our --guardian-type enum spells it "diffusion"; GORMPO/LEQ call the same estimator "ddpm".
guardian_type_of() { [ "$1" = ddpm ] && echo diffusion || echo "$1"; }

# estimator -> GPU (see header for the memory/time reasoning).
declare -A GPU_FOR=( [neuralode]=6 [kde]=7 [vae]=7 [ddpm]=7 )

# Canonical (non-per-seed) guardian checkpoints -- same convention combo_dbg_mcs.sh already
# uses for realnvp (GUARDIAN=.../realnvp/abiomed_realnvp). LEQ_DBG_MCS.sh instead uses
# trained_<type>_<seed>/ guardians (its own per-seed sweep); both are real, differently-
# trained checkpoints on the same real MCS data -- this picks the one already established
# in this repo. Override GUARDIAN_BASE to point at the trained_<type>_<seed>/ set instead.
GUARDIAN_BASE="${GUARDIAN_BASE:-/public/gormpo/models/abiomed}"
guardian_path() {
  case "$1" in
    kde)       echo "$GUARDIAN_BASE/kde/trained_kde_1" ;;
    vae)       echo "$GUARDIAN_BASE/vae/abiomed_vae" ;;
    ddpm)      echo "$GUARDIAN_BASE/diffusion" ;;
    neuralode) echo "$GUARDIAN_BASE/neuralODE/" ;;  # trailing slash required -- see run_combo_dbg_mcs.py
  esac
}

# 0 (true) if the guardian at $2 (path) for type $1 exists. Mirrors LEQ_DBG_MCS.sh's
# guardian_exists(), same file-suffix convention per type.
guardian_exists() {
  case "$1" in
    kde)       [ -f "${2}.faiss" ] && [ -f "${2}_metadata.pkl" ] ;;
    vae)       [ -f "${2}_model.pth" ] && [ -f "${2}_meta_data.pkl" ] ;;
    neuralode) [ -f "${2}_model.pt" ] && [ -f "${2}_metadata.pkl" ] ;;
    ddpm)      [ -f "${2}/checkpoint.pt" ] ;;
  esac
}

# The dynamics ensemble depends on (task, dataset, seed) only -- the guardian is applied in
# EnsembleDynamics.step() at rollout time, never in dynamics.train() -- so one trained
# ensemble serves every estimator here too (same reuse combo_dbg_all_estimators_sparse.sh
# does for its 3 tasks). DYNAMICS_DIR=<path> pins a specific one; if none is found, each
# job below just trains its own (slower, but still correct).
find_dynamics() {
  local d
  if [ -n "${DYNAMICS_DIR:-}" ]; then echo "$DYNAMICS_DIR"; return; fi
  while IFS= read -r d; do
    [ -f "$d/dynamics.pth" ] && { echo "$d"; return; }
  done < <(ls -dt "log/abiomed/combo/seed_${SEED}&timestamp_"*"/model" 2>/dev/null || true)
}
DYN_DIR="$(find_dynamics)"

# preflight: fail loudly now, not partway into a multi-hour run
for est in $ESTIMATORS; do
  g="$(guardian_path "$est")"
  guardian_exists "$est" "$g" || { echo "ERROR: guardian not found for $est at $g" >&2; exit 1; }
  penalty_coef "$est" >/dev/null || exit 1
  [ -n "${GPU_FOR[$est]:-}" ] || { echo "ERROR: no GPU assigned for estimator '$est'" >&2; exit 1; }
done
if [ -n "$DYN_DIR" ]; then
  echo "preflight OK: $(echo $ESTIMATORS | wc -w) estimators, seed $SEED, reusing dynamics: $DYN_DIR"
else
  echo "preflight OK: $(echo $ESTIMATORS | wc -w) estimators, seed $SEED, NO prior dynamics found -- each job trains its own"
fi

# Detach from the terminal/SSH session so training survives it closing -- same pattern as
# combo_dbg_all_estimators_sparse.sh. DETACH=0 to stay in the foreground.
if [ "${DETACH:-1}" = 1 ] && [ -z "${DBG_DETACHED:-}" ]; then
  mkdir -p "$REPO/log/_runs"
  LOGFILE="$REPO/log/_runs/dbg_mcs_$(echo $ESTIMATORS | tr ' ' '-')_$(date +%m%d-%H%M%S).log"
  DBG_DETACHED=1 setsid nohup bash "$SELF" "$@" >"$LOGFILE" 2>&1 </dev/null &
  echo "detached: pid $!"
  echo "  tail -f $LOGFILE"
  exit 0
fi

RESULTS_DIR="log/combo_dbg_mcs_estimators"
mkdir -p "$RESULTS_DIR" "$REPO/log/_runs"
SUMMARY="$RESULTS_DIR/summary_seed${SEED}_$(date +%m%d-%H%M%S).csv"
echo "estimator,penalty_coef,n_episodes,return,sem" > "$SUMMARY"

run_one() {
  local est="$1" gpu="${GPU_FOR[$1]}" g gtype penalty logf dyn_args=() chunk_args=()
  g="$(guardian_path "$est")"; gtype="$(guardian_type_of "$est")"; penalty="$(penalty_coef "$est")"
  logf="$REPO/log/_runs/dbg_mcs_${est}_$(date +%m%d-%H%M%S).log"
  [ "$est" = neuralode ] && chunk_args=(--guardian-chunk-size "$NEURALODE_CHUNK")
  [ -n "$DYN_DIR" ] && dyn_args=(--load-dynamics-path "$DYN_DIR")

  (
    set -euo pipefail
    echo ">>> COMBO+DBG MCS est=$est gpu=$gpu penalty=${PENALTY_TYPE}x${penalty} guardian=$g"

    CUDA_VISIBLE_DEVICES="$gpu" python "$SCRIPT" \
      ${dyn_args[@]+"${dyn_args[@]}"} \
      --classifier-path "$g" \
      --guardian-type "$gtype" \
      --penalty-coef "$penalty" \
      --penalty-type "$PENALTY_TYPE" \
      --seed "$SEED" --device cuda \
      ${chunk_args[@]+"${chunk_args[@]}"}

    # run_combo_dbg_mcs.py bakes guardian_type into the log dir (make_log_dirs(...,
    # record_params=["guardian_type"])), so this glob can only ever match runs of THIS
    # estimator -- no race with the other estimators launched alongside it, unlike a bare
    # seed+timestamp path shared by everyone.
    mydir="$(ls -dt "log/abiomed/combo&guardian_type=${gtype}/seed_${SEED}&timestamp_"* 2>/dev/null | head -1 || true)"
    [ -n "$mydir" ] || { echo "ERROR: no log dir found for $est (guardian_type=$gtype)" >&2; exit 1; }
    [ -f "$mydir/model/policy.pth" ] || { echo "ERROR: no policy.pth for $est at $mydir" >&2; exit 1; }
    echo "    log dir: $mydir"

    echo ">>> Evaluating $est on held-out test windows"
    row=$(CUDA_VISIBLE_DEVICES="$gpu" python run_example/eval_combo_mcs.py \
            --policy-path "$mydir/model/policy.pth" \
            --eval_episodes "$EVAL_EPISODES" --seed "$EVAL_SEED" --device cuda \
          | grep '^CSVROW,' || true)
    [ -n "$row" ] || { echo "ERROR: eval failed for $est" >&2; exit 1; }
    echo "${est},${penalty},${row#CSVROW,}" >> "$SUMMARY"
    echo "OK $est -> $(tail -1 "$SUMMARY")"
  ) > "$logf" 2>&1 &
}

for est in $ESTIMATORS; do
  run_one "$est"
  sleep 10   # gentle stagger on shared setup I/O (loading the same real-data pickle); not
             # needed for correctness -- guardian_type in the log dir path already rules
             # out cross-estimator collisions regardless of timing.
done
wait

echo "All estimators done."
echo "Summary: $REPO/$SUMMARY"
cat "$SUMMARY"
