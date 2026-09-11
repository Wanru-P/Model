#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

python - <<'PY'
import sys
if sys.version_info[:2] != (3, 10):
    raise SystemExit(
        "E1 requires Python 3.10, but found {}.{}.".format(
            sys.version_info.major, sys.version_info.minor))
print("Python:", sys.version.split()[0])
PY

python -m pip install --upgrade pip wheel
python -m pip install --no-cache-dir -r requirements.txt
python -m pip check

python - <<'PY'
import paddle
print("Paddle:", paddle.__version__)
print("Compiled CUDA:", paddle.version.cuda())
print("CUDA build:", paddle.is_compiled_with_cuda())
PY

echo "Environment installation completed."
echo "Next: place the dataset and pretrained weights, then run:"
echo "  python scripts/check_e1_ready.py"
