#!/usr/bin/env bash
# Run dynamic acceptance only. This script intentionally has no full-training command.
set -Eeuo pipefail
if [[ $# -ne 1 ]]; then echo "Usage: bash scripts/run_e6a_dynamic_acceptance.sh /path/to/E1_best.pdparams" >&2; exit 2; fi
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1
python scripts/check_e6a_static.py
nvidia-smi
mkdir -p output/E6a_RGBIR_DAMSDet_960_dual_o2m
python -u tools/e6a_dynamic_acceptance.py --checkpoint "$1" 2>&1 | tee "output/E6a_RGBIR_DAMSDet_960_dual_o2m/dynamic_acceptance_$(date +%Y%m%d_%H%M%S).log"
