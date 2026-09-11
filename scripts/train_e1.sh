#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export PYTHONUTF8=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

CONFIG="configs/damsdet/damsdet_r50vd_aic2026_e1_rgbir_960_letterbox.yml"
EXPERIMENT_ROOT="output/E1_RGBIR_DAMSDet_960_letterbox"
LOG_DIR="${EXPERIMENT_ROOT}/logs"
mkdir -p "${LOG_DIR}"

RESUME_CHECKPOINT="${1:-}"
RESUME_BEST_AP="${2:-}"
if [[ -n "${RESUME_BEST_AP}" && -z "${RESUME_CHECKPOINT}" ]]; then
    echo "A resume checkpoint is required when resume_best_ap is provided." >&2
    exit 2
fi
if [[ -n "${RESUME_CHECKPOINT}" && ! -f "${RESUME_CHECKPOINT}" ]]; then
    echo "Resume checkpoint does not exist: ${RESUME_CHECKPOINT}" >&2
    exit 2
fi

python scripts/check_e1_ready.py
nvidia-smi || true

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_PATH="${LOG_DIR}/train_${STAMP}.log"
ARGS=(-u tools/train.py -c "${CONFIG}" --eval --amp)
if [[ -n "${RESUME_CHECKPOINT}" ]]; then
    ARGS+=(-r "${RESUME_CHECKPOINT}")
fi
if [[ -n "${RESUME_BEST_AP}" ]]; then
    ARGS+=(-o "resume_best_ap=${RESUME_BEST_AP}")
fi

echo "Starting E1 in the foreground."
echo "Live output and log: ${LOG_PATH}"
python "${ARGS[@]}" 2>&1 | tee "${LOG_PATH}"
