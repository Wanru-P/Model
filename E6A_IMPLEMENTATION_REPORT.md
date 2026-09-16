# E6a implementation report — Sol redo

Current verdict: **NOT READY**. Dynamic acceptance has not been run.

## A Git

- Branch: `codex/e6a-sol-redo`
- Base HEAD: `40e0066642ca9370f594fa25006dd43623f093cf`
- Sol implementation commit subject:
  `Reimplement E6a dual O2M from verified E1 baseline`.
- Baseline evidence: E1 960 letterbox config, dual ResNet50-vd, dual
  HybridEncoder, DAMS transformer, 300 queries, six decoder layers.
- Status before: clean independent worktree at E1 base commit.
- Status after: clean Sol worktree after the implementation commit.

Static handoff status: five existing production files are minimally modified;
three official upstream runtime files, two E6a configs, three scripts, one
acceptance tool, and two regression tests are added. Python syntax and
`git diff --check` pass. CUDA/runtime checks remain intentionally unexecuted.

## B Terra leftovers

Pre-existing Terra E6a modifications are isolated on `main` commit `1014788`.
They were not deleted, overwritten, cherry-picked, or used as implementation
source. The commit contains its own configs, transform edits, architecture
edits, diagnostics, and report. Sol was reimplemented from E1 commit `40e0066`.

## C Upstream sources

Source: `PaddlePaddle/PaddleDetection` commit
`b25522a0f4bde8c80603f3ba5e3472059972e3b5`.

| Source/local file | SHA-256 | Compatibility change |
| --- | --- | --- |
| `ppdet/modeling/heads/ppyoloe_head.py` | `2BDB1457A71C2280267CF87D73ECBB8FCEA5559806825048B90F918B03EC0035` | None; verbatim official file. |
| `ppdet/modeling/assigners/atss_assigner.py` | `0F725D402C42F518D463FC0EB6CD91BF2BFD86889B9830DD73F2331A4450368B` | None; verbatim official file. |
| `ppdet/modeling/assigners/task_aligned_assigner.py` | `4CED1D0A06EE48B29E0A5B5FE057320EAA962809078AE0076636BBEECF978C36` | None; verbatim official file. |

Project compatibility is limited to package registration, two isolated origin
GT transforms, and optional dual-head wiring in DAMSDet.

The E1-base Hungarian matcher file is unchanged byte-for-byte by Sol. The
requested AMP stability behavior is therefore not reverted or otherwise
altered; no matcher patch is part of E6a.

## D Architecture

VIS backbone → VIS HybridEncoder P3/P4/P5 → DAMS main path + independent VIS
PPYOLOE head. IR follows the same layout with a separate PPYOLOE head. The DAMS
transformer and DINO head remain the E1 implementation. Auxiliary heads run
only while `self.training` is true.

## E Parameter count

Pending cloud dynamic acceptance: E1 total, E6a total, VIS aux, IR aux.

## F GT pipeline

Implemented as geometry/resize/letterbox → `Multi_PreserveOriginGT` pixel XYXY
copy → unchanged `Multi_NormalizeBox` → unchanged `BboxXYXY2XYWH` → DINO
normalized CXCYWH. `PadOriginGT` pads only auxiliary tensors. Real-sample
roundtrip evidence is pending.

## G E1 equivalence

Pending cloud dynamic acceptance for both backbones, both necks, decoder
logits/boxes, final scores/boxes, and `bbox_num`.

## H Aux disabled train regression

Pending cloud dynamic acceptance for every original DINO loss and total.

## I Assigner

Configured: epoch 0/29 ATSS; epoch 30/31 TaskAligned. Runtime evidence pending.

## J Positive counts

Pending at least 20 real train batches, including person/sign/ball/uav.

## K Gradient

Pending DINO-only, AUX-only, and total gradient norms for VIS/IR backbone and
neck, auxiliary heads, and auxiliary-to-DINO neck gradient ratios.

## L Tiny-overfit

Pending the pre-existing 24-image train subset. No replacement subset will be
invented by this implementation.

## M 100-step smoke

Pending CSV with losses, ratio distribution, gradient norms, LR, and peak GPU
memory in Paddle 2.6.2/CUDA 11.8.

## N Eval

Pending auxiliary call counts and 10-warmup/30-measure latency comparison.

## O Weight loading

E1 best SHA-256:
`501AC0DE64A86A809BD67C3D2A23ACD3753B67A2EAAFD23CC98B2C95642BECE2`.
It is used only for equivalence. Formal tiny/smoke/early training starts from
the same COCO pretrained weights as E1; loaded/missing/unexpected/shape
mismatch evidence is pending.

## P Final verdict

**NOT READY**
