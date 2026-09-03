#!/usr/bin/env bash
# COMBO + density guardian (DBG) with the VAE estimator (variational autoencoder ELBO),
# across halfcheetah / hopper / walker2d sparse, seed 42.
#
# Produces the COMBO-VAE policy the t-SNE panel of that name needs. Thin wrapper over
# combo_dbg_all_estimators_sparse.sh -- all config (datasets, guardian paths, per-estimator
# penalty coefficients, rollout-length, cql-weight) lives there.
#
# Cost: ~18 h/task (guardian cost negligible), x3 tasks run sequentially.
#
# Usage:
#   DEVICE=cuda:4 bash $(basename "$0")
set -euo pipefail
exec env ESTIMATORS="vae" bash "$(dirname "${BASH_SOURCE[0]}")/combo_dbg_all_estimators_sparse.sh" "$@"
