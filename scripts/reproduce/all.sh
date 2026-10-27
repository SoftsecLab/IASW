#!/usr/bin/env bash
set -Eeuo pipefail

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${_SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

MODE="${1:---paper-only}"

case "${MODE}" in
    --paper-only)
        echo "======================================================================"
        echo "IASW reviewer reproduction: PAPER-ONLY"
        echo "======================================================================"
        bash scripts/reproduce/00_check.sh
        bash scripts/reproduce/50_analysis.sh
        ;;

    --full)
        echo "======================================================================"
        echo "IASW reviewer reproduction: FULL"
        echo "======================================================================"
        bash scripts/reproduce/00_check.sh
        bash scripts/reproduce/30_main.sh
        bash scripts/reproduce/40_ablations.sh
        bash scripts/reproduce/50_analysis.sh
        ;;

    *)
        echo "Usage:"
        echo "  bash scripts/reproduce/all.sh --paper-only"
        echo "  bash scripts/reproduce/all.sh --full"
        exit 1
        ;;
esac

echo
echo "======================================================================"
echo "REPRODUCTION WORKFLOW COMPLETE"
echo "======================================================================"
