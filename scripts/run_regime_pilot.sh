#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
python3 -m pytest -q
python3 run.py --protocol regime --profile pilot \
  --output results/inferred_regime_pilot
