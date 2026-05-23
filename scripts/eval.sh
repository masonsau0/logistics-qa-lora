#!/usr/bin/env bash
# Evaluate the trained adapter against the base model side-by-side.
#
# Usage:
#   ./scripts/eval.sh                          # default: compare both
#   ./scripts/eval.sh --no-adapter             # base model only
#   ./scripts/eval.sh --adapter path/to/dir    # custom adapter path

set -euo pipefail

cd "$(dirname "$0")/.."

if [ -f .env ]; then
    set -a
    # shellcheck disable=SC1091
    source .env
    set +a
fi

python -m src.evaluate --compare "$@"
