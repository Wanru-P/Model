#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -ne 3 ]]; then
    echo "Usage: bash scripts/run_e6a_acceptance.sh E1_BEST COCO_PRETRAIN TINY24_JSON" >&2
    exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1

python scripts/check_e6a_static.py
python -u tools/e6a_sol_acceptance.py \
  --e1-checkpoint "$1" \
  --coco-pretrain "$2" \
  --tiny-annotation "$3"

python scripts/check_e6a_ready.py
