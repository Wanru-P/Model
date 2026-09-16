# AIC2026 E1：RGB+IR DAMSDet 960 Letterbox

本目录是可提交到 GitHub、并在 Ubuntu 22.04 云 GPU 上复现实验 E1 的独立工程。

E1 使用完整 RGB+IR DAMSDet，不包含 Depth Encoder 或 Depth Gate。训练集/验证集固定为同一份 1600/400 group-aware + class-aware 划分；模型、COCO 预训练权重、优化器、学习率、72 epoch、300 queries、6 层 decoder 与 E0 一致。验证输入从 E0 的 640 改为 960 letterbox。

## 1. 云环境

- Ubuntu 22.04
- Python 3.10
- CUDA 11.8
- PaddlePaddle GPU 2.6.2
- 单卡训练，默认 `CUDA_VISIBLE_DEVICES=0`

`requirements.txt` 使用 Paddle 官方 CUDA 11.8 / CPython 3.10 wheel，不要再安装云平台提供的 Paddle 2.2 或 2.4 环境。

## 2. GitHub 中不包含的大文件

普通 GitHub 仓库不能保存本实验的完整数据集，也不能直接保存 164 MiB 的预训练权重。克隆仓库后，在云服务器上放成以下结构：

```text
data/AIC2026_Train_2000/
├── visible/       # 2000 张 RGB
├── infrared/      # 2000 张 IR
└── depth/         # 2000 张 Depth；为保持 E0 reader 完全一致仍需提供

weights/
└── coco_pretrain_weights.pdparams
```

固定的 `train.json` 和 `val.json` 已经包含在仓库中，无需重新划分数据。

## 3. 安装环境

```bash
git clone <你的 GitHub 仓库地址>
cd AIC2026_E1_GitHub

nvidia-smi
python --version
bash scripts/setup_cloud.sh
```

安装完成并放好数据与权重后执行：

```bash
python scripts/check_e1_ready.py
```

只有看到 `Cloud E1 readiness check passed` 后才开始正式训练。

脚本会固定 `OMP_NUM_THREADS=1`。AutoDL 默认值可能是 16，Paddle 会对此发出警告；它与数据缺失不是同一个问题。

## 4. 正式训练 E1

```bash
bash scripts/train_e1.sh
```

训练在前台运行，终端会持续显示 epoch、loss、learning rate 和验证 mAP，同时日志保存到：

```text
output/E1_RGBIR_DAMSDet_960_letterbox/logs/
```

模型 checkpoint 位于：

```text
output/E1_RGBIR_DAMSDet_960_letterbox/
└── damsdet_r50vd_aic2026_e1_rgbir_960_letterbox/
```

训练命令已启用 `--eval --amp`，会按照配置周期运行 COCO 验证并输出 mAP@50-95、AP50、AP75、AP-small/medium/large 和每类 AP。

## 5. 中断后继续训练

第一个参数是 `.pdparams` checkpoint，第二个参数是中断前日志中的 best mAP@50-95：

```bash
bash scripts/train_e1.sh \
  output/E1_RGBIR_DAMSDet_960_letterbox/damsdet_r50vd_aic2026_e1_rgbir_960_letterbox/47.pdparams \
  0.443
```

checkpoint 文件名以服务器实际保存的 epoch 为准，`0.443` 也必须替换成 E1 中断前真实的 best mAP。

## 6. 单独验证某个 checkpoint

```bash
bash scripts/eval_e1.sh \
  output/E1_RGBIR_DAMSDet_960_letterbox/damsdet_r50vd_aic2026_e1_rgbir_960_letterbox/best_model.pdparams
```

如果实际最优文件不是 `best_model.pdparams`，替换为对应路径即可。

## 7. 实验解释注意事项

现有 E0 实际使用 `keep_ratio: false` 的 640 方形拉伸，而本 E1 按要求使用 960 letterbox。因此 E0→E1 同时改变了分辨率和 resize 几何策略。当前实验仍可判断这一整套高分辨率 letterbox 方案是否涨分，但不能把全部增益严格归因于分辨率；如果需要纯因果消融，应补跑 640-letterbox 对照。

## 8. 只读诊断：padding Top-300 与提交 Top-100

以下两个命令不训练、不改 production code；每个命令只对固定的 400-val 做一次 `model.eval()` 推理。必须传入 E1 正式 56 分对应的 best checkpoint：

```bash
bash scripts/run_e1_padding_diagnostic.sh /path/to/e1_best.pdparams
bash scripts/run_e1_submission_diagnostic.sh /path/to/e1_best.pdparams
```

第一个命令输出 `output/diagnostics/e1_padding/`，包括逐图 Top-300 padding CSV、六层 memory 来源统计和 encoder score 对比。第二个命令输出 `output/diagnostics/e1_submission/`，包括 raw decoder query、production Stage-2 prediction dump、Top-100 saturation、duplicate、rank 101--300 lost TP、每类 occupancy 与 36 组以内离线 confidence/NMS sweep。两项都完成后自动生成：

```text
output/diagnostics/E1_DIAGNOSTIC_REPORT.md
```

离线 sweep 仅用于决定后续是否值得做提交后处理实验，不会自动修改提交参数或训练配置。
