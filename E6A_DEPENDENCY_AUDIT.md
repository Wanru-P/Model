# E6a dependency audit

Status: **source audit complete; minimal implementation in progress**.  The
initial audit was performed before production-code changes, as required by the
E6a specification.

## Repository state observed

- Branch/HEAD: `main` / `40e0066 解决合并冲突`
- Existing user changes were preserved.  No reset, clean, switch, upgrade, or
  training command was run for E6a.
- The trainer backpropagates exactly `outputs['loss']`; therefore E6a must make
  `loss` the sole optimized aggregate and expose all component losses under
  non-`loss*` logging keys to avoid double counting.
- `Trainer.train` adds `data['epoch_id']` before every forward, so the required
  epoch 0/29 ATSS and epoch 30/31 TaskAligned transition can be connected
  without inventing an epoch source.

## Component matrix

| Component | Exists | Compatible with E6a | Action |
| --- | --- | --- | --- |
| `PPYOLOEHead` | Initially no | Yes after verbatim import | Imported from the named upstream commit. |
| Tensor `ATSSAssigner` | Initially no | Yes after verbatim import | Imported from the named upstream commit. `ppdet/data/transform/atss_assigner.py` remains unused. |
| `TaskAlignedAssigner` | Initially no | Yes after verbatim import | Imported from the named upstream commit. |
| `generate_anchors_for_grid_cell` | Yes | Yes | Existing `ppdet/modeling/assigners/utils.py` provides the Paddle implementation needed by the auxiliary head. |
| `gather_topk_anchors` / inside-box / max-IoU helpers | Yes | Yes | Existing assigner utilities are the expected support layer. |
| `GIoULoss` | Yes | Likely | Existing `ppdet/modeling/losses/iou_loss.py`; exact upstream call signature still needs port-time verification. |
| `ConvBNLayer` / `RepVggBlock` | Yes | Needs import adaptation | Available in `ppdet/modeling/backbones/cspresnet.py`, rather than a head-local module. |
| `MultiClassNMS` | Yes | Not required for E6a train-only path | Available in `ppdet/modeling/layers.py`; it must not be invoked during E6a evaluation. |

## Current E1 call-chain facts

1. `DAMSDet._forward` sends RGB/IR backbone features through separate
   HybridEncoders, then invokes the DAMS transformer and DINO head.
2. The HybridEncoder outputs are three lists of 256-channel stride 8/16/32
   features; their dynamic spatial dimensions are determined by the random
   resize, so E6a must use `neck_*.out_shape` rather than hard-coded shapes.
3. Train transforms are `Multi_BatchRandomResize`, image normalize,
   `Multi_NormalizeBox`, then `BboxXYXY2XYWH`.  Thus current `gt_bbox` is
   normalized CXCYWH when DINO sees it; no `origin_gt_bbox` path exists.
4. `PadGT` exists but has no origin-GT-only option.  A minimal opt-in origin
   padding extension or a separate `PadOriginGT` is required.
5. Evaluation must not call either auxiliary head.  E6a can preserve E1 eval
   equivalence if the heads are reached only inside `self.training`.

## Upstream-source status

The requested source repository and commit are:

`PaddlePaddle/PaddleDetection@b25522a0f4bde8c80603f3ba5e3472059972e3b5`

The initial audit could not retrieve GitHub blobs. The user subsequently
provided the official source tree locally at
`C:\Users\Admin\Desktop\PaddleDetection_E6a_upstream_b25522`. The three
required runtime files were inspected line-by-line and copied verbatim, with
their SHA-256 values recorded in `E6A_COMPATIBILITY.md`. The supplied RT-DETRv3
YAML files were inspected only to verify the O2M recipe; they were not merged
into DAMSDet. No PPYOLOE/assignment code was reconstructed from memory.

## Stop condition

Source acquisition is no longer a blocker. Runtime acceptance remains required
before E6a can be declared ready; see `E6A_IMPLEMENTATION_REPORT.md`.
