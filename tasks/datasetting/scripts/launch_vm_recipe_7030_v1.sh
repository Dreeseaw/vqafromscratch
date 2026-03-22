#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

export STAGE1_RATIO_PERCENT="${STAGE1_RATIO_PERCENT:-70}"
export STAGE2_RATIO_PERCENT="${STAGE2_RATIO_PERCENT:-30}"
export STAGE3_RATIO_PERCENT="${STAGE3_RATIO_PERCENT:-0}"

exec "${SCRIPT_DIR}/launch_vm_recipe_ratio_v1.sh" "$@"
