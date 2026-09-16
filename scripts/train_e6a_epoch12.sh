#!/usr/bin/env bash
# Explicit early-run command. It refuses to start until dynamic acceptance passes.
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1

python scripts/check_e6a_ready.py
mkdir -p output/E6a_Sol_RGBIR_DAMSDet_960_dual_o2m/logs
python -u tools/train.py \
  -c configs/damsdet/damsdet_r50vd_aic2026_e6a_sol_960_dual_o2m.yml \
  --eval --amp -o epoch=12 2>&1 | \
  tee "output/E6a_Sol_RGBIR_DAMSDet_960_dual_o2m/logs/epoch12_$(date +%Y%m%d_%H%M%S).log"
