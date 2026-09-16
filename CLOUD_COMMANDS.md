# E6a Sol cloud commands

This repository must be cloned into a directory separate from the existing E1
checkout. The commands below share E1 data and weight directories read-only by
path/symlink; E1 configs, checkpoints, logs, and output are not modified.

## 1. Push the Sol branch from Windows

```powershell
cd "C:\Users\Admin\Desktop\AIC-城市目标检测\AIC2026_E6a_Sol_Redo"
git status --short
git log -2 --oneline
git push -u origin codex/e6a-sol-redo
```

## 2. Create the independent cloud checkout

Replace `REPO_URL` with the repository URL. Stop if `TINY24_JSON` cannot be
resolved to the pre-existing 24-image train subset; do not create a substitute.

```bash
E1_DIR=/root/autodl-tmp/Model
E6_DIR=/root/autodl-tmp/E6a_Sol_Redo
REPO_URL='YOUR_GITHUB_REPOSITORY_URL'
E1_BEST="$E1_DIR/weights/39.pdparams"
COCO_PRETRAIN="$E1_DIR/weights/coco_pretrain_weights.pdparams"
TINY24_JSON="$E1_DIR/dataset/aic2026_qadepth/annotations_stratified_candidate/tiny24_train.json"

test -d "$E1_DIR"
test -f "$E1_BEST"
test -f "$COCO_PRETRAIN"
test -f "$TINY24_JSON"
test ! -e "$E6_DIR"

git clone --branch codex/e6a-sol-redo "$REPO_URL" "$E6_DIR"
cd "$E6_DIR"

for modality in visible infrared depth; do
  test -d "$E1_DIR/data/AIC2026_Train_2000/$modality"
  rm -f "data/AIC2026_Train_2000/$modality/.gitkeep"
  rmdir "data/AIC2026_Train_2000/$modality"
  ln -s "$E1_DIR/data/AIC2026_Train_2000/$modality" \
    "data/AIC2026_Train_2000/$modality"
done

rm -f weights/.gitkeep
rmdir weights
ln -s "$E1_DIR/weights" weights

python -c "import paddle; print(paddle.__version__, paddle.is_compiled_with_cuda())"
```

The final command must print `2.6.2 True`.

## 3. Run acceptance only

```bash
cd /root/autodl-tmp/E6a_Sol_Redo
bash scripts/run_e6a_acceptance.sh \
  /root/autodl-tmp/Model/weights/39.pdparams \
  /root/autodl-tmp/Model/weights/coco_pretrain_weights.pdparams \
  /root/autodl-tmp/Model/dataset/aic2026_qadepth/annotations_stratified_candidate/tiny24_train.json
```

Success requires both:

```text
E6a readiness check passed
READY FOR E6a EARLY TRAINING
```

No epoch training is started by the acceptance command.

## 4. Start only the epoch-12 early gate after acceptance

```bash
cd /root/autodl-tmp/E6a_Sol_Redo
bash scripts/train_e6a_epoch12.sh
```

This command starts E6a from the same COCO pretrain configured by E1. It does
not load or fine-tune `39.pdparams`, and it stops at epoch 12.
