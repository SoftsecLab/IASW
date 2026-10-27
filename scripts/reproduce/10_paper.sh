#!/usr/bin/env bash
set -Eeuo pipefail

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${_SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

echo "======================================================================"
echo "Reproduce paper outputs from existing raw results"
echo "======================================================================"

[[ -x scripts/paper.sh || -f scripts/paper.sh ]] || {
    echo "ERROR: scripts/paper.sh not found." >&2
    exit 1
}

bash scripts/paper.sh

echo
echo "[Outputs]"
find paper/latex -maxdepth 1 -type f 2>/dev/null | sort || true
find paper/csv -maxdepth 1 -type f 2>/dev/null | sort || true
find paper/figures -maxdepth 1 -type f 2>/dev/null | sort || true
