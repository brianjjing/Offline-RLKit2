#!/usr/bin/env bash
# COMBO + density guardian (DBG) with the DDPM estimator (diffusion ELBO),
# across halfcheetah / hopper / walker2d sparse, seed 42.
#
# Produces the COMBO-DDPM policy the t-SNE panel of that name needs. Thin wrapper over
# combo_dbg_all_estimators_sparse.sh -- all config (datasets, guardian paths, per-estimator
# penalty coefficients, rollout-length, cql-weight) lives there.
#
# Cost: ~19 h/task (0.9 h of it guardian), x3 tasks run sequentially.
#
# Usage:
#   DEVICE=cuda:4 bash $(basename "$0")
set -euo pipefail
exec env ESTIMATORS="diffusion" bash "$(dirname "${BASH_SOURCE[0]}")/combo_dbg_all_estimators_sparse.sh" "$@"
