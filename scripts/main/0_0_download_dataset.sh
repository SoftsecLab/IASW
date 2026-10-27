#!/bin/bash
set -e

# === PROJECT ROOT BOOTSTRAP (refactor v4) ===
_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_PROJECT_SCAN="${_SCRIPT_DIR}"
while [[ "${_PROJECT_SCAN}" != "/" && ! -d "${_PROJECT_SCAN}/src" ]]; do
    _PROJECT_SCAN="$(dirname "${_PROJECT_SCAN}")"
done
if [[ ! -d "${_PROJECT_SCAN}/src" ]]; then
    echo "ERROR: cannot locate project root containing src/" >&2
    exit 1
fi
PROJECT_DIR="${PROJECT_DIR:-${_PROJECT_SCAN}}"
PROJECT_ROOT="${PROJECT_ROOT:-${PROJECT_DIR}}"
SCRIPT_DIR="${SCRIPT_DIR:-${_SCRIPT_DIR}}"
cd "${PROJECT_DIR}"
export PROJECT_DIR PROJECT_ROOT
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
# === END PROJECT ROOT BOOTSTRAP ===

export CUDA_VISIBLE_DEVICES=0
export HF_ENDPOINT=https://hf-mirror.com
export NLTK_DATA=./models/nltk_data
PROJECT_DIR="${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"


python data/download_datasets.py