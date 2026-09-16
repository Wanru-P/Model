# E6a implementation report

Date: 2026-09-16  
Scope: E1 RGB+IR DAMSDet 960 letterbox + training-only VIS/IR PPYOLOE O2M
auxiliary heads, `lambda_aux = 1.0`.

## Implemented

- Three runtime files were copied verbatim from
  `PaddlePaddle/PaddleDetection@b25522a0f4bde8c80603f3ba5e3472059972e3b5`.
  Exact SHA-256, original Apache-2.0 headers, source path, and all local
  compatibility changes are in `E6A_COMPATIBILITY.md`.
- The existing DAMSDet RGB/IR backbones, HybridEncoders, DAMS transformer,
  DINO loss/matcher, 300 queries, six decoder layers, 960 letterbox geometry,
  optimizer/LR schedule, and post-processing are not changed.
- `Multi_NormalizeBox(retain_origin_box=True)` copies the synchronized
  post-geometry/post-letterbox pixel-space XYXY targets before DINO's
  normalization. `PadGT(only_origin_box=True)` creates only the O2M padding
  tensors; it does not alter DINO targets.
- The E6a train config uses upstream ATSS for epoch IDs 0--29 and upstream
  TaskAligned assignment from epoch ID 30. VIS and IR have independent heads.
- The auxiliary heads are invoked exclusively inside `DAMSDet.training`; eval
  and inference take the original DINO-only branch.

## Static acceptance completed

Run:

```powershell
& 'E:\Users\Admin\anaconda3\python.exe' .\scripts\check_e6a_static.py
```

Result: PASS for all three upstream file hashes, Apache headers, Python syntax,
origin-GT reader contract, `lambda_aux=1.0`, and `static_assigner_epoch=30`.

## Cloud dynamic-acceptance runner

`tools/e6a_dynamic_acceptance.py` and
`scripts/run_e6a_dynamic_acceptance.sh` implement the requested finite,
non-epoch acceptance sequence: eval equivalence, disabled-aux DINO regression,
24 genuine train-sample GT roundtrips, assignment transition hooks,
auxiliary-only gradients, 24-sample tiny-overfit, 100 real-train AMP steps,
and a save/reload/one-step checkpoint test. The runner has no 72-epoch path.
It is deliberately not marked executed until its JSON report exists from the
original E1 cloud runtime.

## Runtime acceptance status

| Required acceptance item | Status | Evidence / blocker |
| --- | --- | --- |
| E1 inference exact equivalence | NOT RUN | The local Python environment cannot read the prior Paddle dependency folder (`Access denied` on `.gpu_deps/paddle/__init__.py`). |
| Aux-disabled train DINO regression | NOT RUN | Same unavailable Paddle runtime. |
| Pixel XYXY ↔ normalized CXCYWH roundtrip | IMPLEMENTED, NOT RUN | Reader preserves XYXY before the existing normalization/conversion; needs tensor/data run. |
| ATSS epochs 0/29 and TAL epochs 30/31 | CONFIGURED, NOT RUN | `PPYOLOEHead` upstream branch uses `epoch_id < 30`; needs runtime instrumentation. |
| AUX-only VIS/IR gradient check | NOT RUN | Same unavailable Paddle runtime. |
| Tiny-overfit | NOT RUN | Current repository has 0/1600 images from `train.json`; only val-400 files are present. |
| 100-step AMP smoke | NOT RUN | Same unavailable runtime and missing train-1600 images. |
| Eval auxiliary-forward count = 0 | IMPLEMENTED, NOT RUN | Control flow excludes heads when `self.training` is false; needs hook-based run. |

## Required next environment before formal training

1. Restore/provide all 1,600 `train.json` RGB/IR/Depth triplets under the
   configured `data/AIC2026_Train_2000` tree.
2. Use the intended cloud environment (PaddlePaddle 2.6.x, Python 3.10,
   CUDA 11.8) or restore a readable local Paddle environment.
3. Run all runtime acceptance checks, including the requested 100-step AMP
   smoke, before launching any staged E6a training.

## Final status

**NOT READY FOR FULL E6a TRAINING.** The implementation and static provenance
checks are complete, but required runtime checks have not been executed and
must not be claimed as passed. No full training was started.
