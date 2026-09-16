#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: bash scripts/run_e1_padding_diagnostic.sh PATH_TO_E1_BEST.pdparams" >&2
    exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
CHECKPOINT="$1"
CONFIG="configs/damsdet/damsdet_r50vd_aic2026_e1_rgbir_960_letterbox.yml"
OUTPUT="output/diagnostics/e1_padding"
cd "${REPO_ROOT}"

if [[ ! -f "${CHECKPOINT}" ]]; then
    echo "Checkpoint does not exist: ${CHECKPOINT}" >&2
    exit 2
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export PYTHONUTF8=1
export OMP_NUM_THREADS=1

mkdir -p "${OUTPUT}"
{
    echo "captured_at_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "--- git log -1 ---"
    git log -1 --oneline
    echo "--- git status --short ---"
    git status --short
    echo "--- git diff --stat ---"
    git diff --stat
    echo "--- git diff ---"
    git diff --no-ext-diff
} > "${OUTPUT}/repository_state_before.txt"

echo "repo root: ${REPO_ROOT}"
echo "git HEAD: $(git log -1 --oneline)"
echo "config: ${CONFIG}"
echo "checkpoint: $(realpath "${CHECKPOINT}")"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "output: ${OUTPUT}"

python -u tools/e1_padding_diagnostic.py \
  --config "${CONFIG}" \
  --checkpoint "${CHECKPOINT}" \
  --output-dir "${OUTPUT}" \
  --device gpu:0

{
    echo "captured_at_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "--- git log -1 ---"
    git log -1 --oneline
    echo "--- git status --short ---"
    git status --short
    echo "--- git diff --stat ---"
    git diff --stat
    echo "--- git diff ---"
    git diff --no-ext-diff
} > "${OUTPUT}/repository_state_after.txt"
