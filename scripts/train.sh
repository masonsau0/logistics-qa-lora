#!/usr/bin/env bash
# Train the LoRA adapter. Defaults are tuned for a Colab T4 (16 GB VRAM)
# with QLoRA. Override any flag — they all forward to src/train.py.
#
# Usage:
#   ./scripts/train.sh                       # default 3-epoch run
#   ./scripts/train.sh --epochs 5 --lr 3e-4  # custom hyperparameters

set -euo pipefail

cd "$(dirname "$0")/.."

# Load .env if present (so WANDB_API_KEY etc. are available).
if [ -f .env ]; then
    set -a
    # shellcheck disable=SC1091
    source .env
    set +a
fi

python -m src.train "$@"
