# E6a compatibility and provenance record

## Upstream source audit

The following files were copied verbatim from `PaddlePaddle/PaddleDetection`
commit `b25522a0f4bde8c80603f3ba5e3472059972e3b5`, supplied locally in
`C:\Users\Admin\Desktop\PaddleDetection_E6a_upstream_b25522`.  Their Apache-2.0
copyright and license headers are retained.

| Upstream path | SHA-256 |
| --- | --- |
| `ppdet/modeling/heads/ppyoloe_head.py` | `2BDB1457A71C2280267CF87D73ECBB8FCEA5559806825048B90F918B03EC0035` |
| `ppdet/modeling/assigners/atss_assigner.py` | `0F725D402C42F518D463FC0EB6CD91BF2BFD86889B9830DD73F2331A4450368B` |
| `ppdet/modeling/assigners/task_aligned_assigner.py` | `4CED1D0A06EE48B29E0A5B5FE057320EAA962809078AE0076636BBEECF978C36` |

The two upstream RT-DETRv3 YAML files were inspected as recipe references only;
they are **not** copied or included in this project because they would change
the DAMSDet architecture/reader.

## Minimal project-side changes

1. Import the three upstream modules from the existing heads/assigners package
   initializers so the existing workspace registry can create them.
2. Add optional `retain_origin_box` to `Multi_NormalizeBox`.  It copies the
   synchronized, post-geometry/post-letterbox pixel-space XYXY boxes before
   the established DINO normalization changes `gt_bbox`.
3. Add optional `PadGT(only_origin_box=True)`.  It pads only those copied
   `origin_*` targets and leaves DINO's `gt_bbox`, `gt_class`, and all E1
   behavior unchanged.
4. Add two optional PPYOLOE heads and the weighted training-only loss to
   `DAMSDet`.  The heads are built only when `aux_o2m_enabled: true`, are called
   only from the training branch, and never enter the evaluation/inference
   branch.
5. Add a dedicated E6a config and reader.  All E1 optimization, augmentation,
   960 letterbox, DINO, 300-query, six-decoder-layer, transformer, and
   post-processing settings are retained.

## Compatibility decisions

- This project uses the existing workspace factory. `PPYOLOEHead` has no
  `from_config`, so `in_channels: [256, 256, 256]` is supplied explicitly in
  E6a rather than relying on the ignored `input_shape` argument. These are the
  existing E1 HybridEncoder P3/P4/P5 output channels.
- E6a uses upstream `ATSSAssigner(topk=9)` for epochs `0..29` and upstream
  `TaskAlignedAssigner(topk=13, alpha=1, beta=6)` for epochs `30+`, exactly as
  selected by upstream `PPYOLOEHead.static_assigner_epoch: 30`.
- `origin_gt_bbox` is pixel-space XYXY. Existing `gt_bbox` remains normalized
  CXCYWH after `Multi_NormalizeBox` then `BboxXYXY2XYWH`, exactly as DINO
  expects. PPYOLOE's native `origin_gt_*` branch consumes the former.
- The existing local Paddle 3.x `DETRPostProcess` integer/float issue is not
  patched in this E6a implementation. Any diagnostic adapter used on that
  environment must be external to production code and reported separately.
- No DINO matcher, transformer, query count, Depth path, NWD, padding-aware
  TopK, resolution, post-processing, learning-rate schedule, or global
  PaddleDetection version was changed.
