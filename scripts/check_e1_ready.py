#!/usr/bin/env python3
"""Validate the cloud environment and every local asset required by E1."""

import importlib
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_PADDLE = "2.6.2"
EXPECTED_CUDA = "11.8"


def fail(message):
    raise SystemExit("ERROR: " + message)


def read_images(annotation_path):
    with annotation_path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    return {item["file_name"] for item in payload["images"]}, payload


def main():
    if sys.version_info[:2] != (3, 10):
        fail("Python 3.10 is required; found {}.{}.".format(
            sys.version_info.major, sys.version_info.minor))

    modules = {
        "paddle": "paddle",
        "cv2": "opencv-python-headless",
        "numpy": "numpy",
        "scipy": "scipy",
        "PIL": "Pillow",
        "yaml": "PyYAML",
        "pycocotools": "pycocotools",
        "terminaltables": "terminaltables",
    }
    loaded = {}
    for module_name, package_name in modules.items():
        try:
            loaded[module_name] = importlib.import_module(module_name)
        except Exception as exc:
            fail("cannot import {} ({}): {}".format(
                module_name, package_name, exc))

    paddle = loaded["paddle"]
    if paddle.__version__ != EXPECTED_PADDLE:
        fail("expected Paddle {}, found {}".format(
            EXPECTED_PADDLE, paddle.__version__))
    if not paddle.is_compiled_with_cuda():
        fail("Paddle is not a CUDA build")
    compiled_cuda = str(paddle.version.cuda())
    if not compiled_cuda.startswith(EXPECTED_CUDA):
        fail("expected Paddle CUDA {}, found {}".format(
            EXPECTED_CUDA, compiled_cuda))
    if paddle.device.cuda.device_count() < 1:
        fail("no CUDA GPU is visible to Paddle")
    paddle.set_device("gpu:0")

    annotation_root = (
        ROOT / "dataset" / "aic2026_qadepth" /
        "annotations_stratified_candidate")
    train_names, train_payload = read_images(annotation_root / "train.json")
    val_names, val_payload = read_images(annotation_root / "val.json")
    expected_names = train_names | val_names
    if len(train_names) != 1600 or len(val_names) != 400:
        fail("the fixed split must contain 1600 train and 400 val images")
    if len(expected_names) != 2000:
        fail("train/val image sets overlap or do not cover 2000 unique images")
    if len(train_payload["categories"]) != 12:
        fail("train annotation does not contain 12 categories")
    if len(val_payload["categories"]) != 12:
        fail("val annotation does not contain 12 categories")

    data_root = ROOT / "data" / "AIC2026_Train_2000"
    for modality in ("visible", "infrared", "depth"):
        folder = data_root / modality
        if not folder.is_dir():
            fail("missing directory: {}".format(folder))
        missing = [name for name in expected_names if not (folder / name).is_file()]
        if missing:
            fail("{} is missing {} files; examples: {}".format(
                folder, len(missing), ", ".join(sorted(missing)[:5])))

    checkpoint = ROOT / "weights" / "coco_pretrain_weights.pdparams"
    if not checkpoint.is_file():
        fail("missing pretrained weights: {}".format(checkpoint))
    if checkpoint.stat().st_size < 100 * 1024 * 1024:
        fail("pretrained checkpoint is unexpectedly small: {}".format(
            checkpoint))

    subprocess.run(
        [sys.executable, "-u", "tools/verify_e1_config.py"],
        cwd=str(ROOT),
        check=True)

    try:
        gpu_name = paddle.device.cuda.get_device_name(0)
    except Exception:
        gpu_name = "GPU 0"
    print("Cloud E1 readiness check passed:")
    print("  Python       : {}".format(sys.version.split()[0]))
    print("  Paddle/CUDA  : {} / {}".format(paddle.__version__, compiled_cuda))
    print("  GPU          : {}".format(gpu_name))
    print("  split        : 1600 train / 400 val")
    print("  modalities   : 2000 RGB + 2000 IR + 2000 Depth")
    print("  pretrained   : {:.1f} MiB".format(
        checkpoint.stat().st_size / (1024.0 * 1024.0)))


if __name__ == "__main__":
    main()
