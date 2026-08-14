#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
python3 -m pytest -q
python3 run.py --protocol minigrid-recurrent --profile pilot \
  --output results/minigrid_recurrent_pilot
