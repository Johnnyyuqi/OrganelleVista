#!/usr/bin/env bash
set -euo pipefail
export OUTPUT_DIR="${OUTPUT_DIR:-outputs/debug_$(date +%Y%m%d_%H%M%S)}"
exec bash "$(dirname "${BASH_SOURCE[0]}")/train.sh" --debug_steps 2 --gradient_accumulation_steps 1 "$@"
