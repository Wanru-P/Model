#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: bash scripts/eval_e1.sh PATH_TO_CHECKPOINT.pdparams" >&2
    exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

CHECKPOINT="$1"
if [[ ! -f "${CHECKPOINT}" ]]; then
    echo "Checkpoint does not exist: ${CHECKPOINT}" >&2
    exit 2
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export PYTHONUTF8=1
export OMP_NUM_THREADS=1

CONFIG="configs/damsdet/damsdet_r50vd_aic2026_e1_rgbir_960_letterbox.yml"
LOG_DIR="output/E1_RGBIR_DAMSDet_960_letterbox/logs"
mkdir -p "${LOG_DIR}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_PATH="${LOG_DIR}/eval_${STAMP}.log"

python -u tools/eval.py \
    -c "${CONFIG}" \
    --classwise \
    --amp \
    -o "weights=${CHECKPOINT}" 2>&1 | tee "${LOG_PATH}"
