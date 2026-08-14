#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PHASE="${1:-all}"
cd "$ROOT"

case "$PHASE" in
  test)
    python3 -m pytest -q
    ;;
  experiments)
    python3 run.py --protocol recurrent-tabular --profile pilot \
      --output results/recurrent_tabular_pilot
    python3 run.py --protocol minigrid-recurrent --profile pilot \
      --output results/minigrid_recurrent_pilot
    ;;
  assets)
    python3 scripts/generate_regime_rfe_assets.py
    ;;
  paper)
    python3 scripts/generate_regime_rfe_assets.py
    tectonic paper/regime_rfe_submission.tex --keep-logs
    ;;
  all)
    python3 -m pytest -q
    python3 run.py --protocol recurrent-tabular --profile pilot \
      --output results/recurrent_tabular_pilot
    python3 run.py --protocol minigrid-recurrent --profile pilot \
      --output results/minigrid_recurrent_pilot
    python3 scripts/generate_regime_rfe_assets.py
    tectonic paper/regime_rfe_submission.tex --keep-logs
    ;;
  *)
    echo "Usage: $0 {test|experiments|assets|paper|all}" >&2
    exit 2
    ;;
esac
