#!/usr/bin/env bash
set -Eeuo pipefail

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${_SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

echo "======================================================================"
echo "IASW reproduction check"
echo "Project: ${PROJECT_ROOT}"
echo "======================================================================"

fail=0

require_file() {
    local f="$1"
    if [[ -f "${f}" ]]; then
        echo "[OK] ${f}"
    else
        echo "[MISSING] ${f}"
        fail=$((fail + 1))
    fi
}

require_dir() {
    local d="$1"
    if [[ -d "${d}" ]]; then
        echo "[OK] ${d}/"
    else
        echo "[MISSING] ${d}/"
        fail=$((fail + 1))
    fi
}

echo
echo "[Python]"
python --version
python - <<'PY'
mods = ["numpy", "pandas", "scipy", "sklearn", "matplotlib"]
failed = []
for mod in mods:
    try:
        __import__(mod)
        print(f"[OK] import {mod}")
    except Exception as exc:
        failed.append((mod, str(exc)))
        print(f"[MISSING] import {mod}: {exc}")
if failed:
    raise SystemExit(1)
PY

echo
echo "[Repository]"
require_dir src
require_dir scripts
require_dir analysis
require_dir results
require_file analysis/paper/build_paper.py
require_file scripts/paper.sh

echo
echo "[Canonical main pipeline]"
require_file scripts/main/0_0_download_dataset.sh
require_file scripts/main/0_1_run_pre_calibration_simmark_semstamp.sh
require_file scripts/main/0_2_build_calib_table.sh
require_file scripts/main/1_run_main_pipeline.sh

echo
echo "[Current result availability]"
for d in \
    results/main_exp/c4/llama3.1-8b-instruct/records \
    results/main_exp/dolly/llama3.1-8b-instruct/records \
    results/main_exp/wiki/llama3.1-8b-instruct/records
do
    if [[ -d "${d}" ]]; then
        n="$(find "${d}" -maxdepth 1 -name '*.jsonl' | wc -l)"
        echo "[OK] ${d}: ${n} jsonl"
    else
        echo "[WARN] no existing results: ${d}"
    fi
done

echo
if [[ "${fail}" -gt 0 ]]; then
    echo "CHECK FAILED: ${fail} required repository item(s) missing."
    exit 1
fi
echo "CHECK PASSED."
