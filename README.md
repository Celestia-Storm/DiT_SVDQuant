# DiT + SVDQuant: 量化与评测

本项目基于 DiT (Diffusion Transformer) 模型，集成了 SVDQuant（基于SVD的量化）方法，并支持对生成图片的 FID/KID/sFID 等指标评测。

## 1. 拉取原有 DiT 代码

本项目以 [facebookresearch/DiT](https://github.com/facebookresearch/DiT) 为基础。你可以直接拉取本仓库，或先拉取原始DiT代码后，合并本项目的SVDQuant相关文件。

```bash
git clone https://github.com/facebookresearch/DiT.git
# 或直接拉取本项目
# git clone <your-repo-url>
cd DiT
```

## 2. 创建虚拟环境

推荐使用 Conda 环境。项目已提供 `environment.yml` 文件：

```bash
conda env create -f environment.yml
conda activate DiT
```

如仅需CPU推理，可移除 `cudatoolkit` 和 `pytorch-cuda` 相关依赖。

## 3. 下载预训练模型权重 & 简单采样

首次采样时会自动下载预训练权重。以 256x256 DiT-XL/2 为例：

```bash
python sample.py --image-size 256 --seed 1
```

如需指定权重路径：

```bash
python sample.py --model DiT-XL/2 --image-size 256 --ckpt pretrained_models/DiT-XL-2-256x256.pt
```

## 4. SVDQuant分解与量化（`models.py` 相关说明）

- `models.py` 文件中新增了 `SVDLinear` 类，实现了基于SVD的线性层分解与量化。
- 提供 `apply_svd_to_dit(model, rank_ratio)` 方法，可将DiT模型中的所有 `nn.Linear` 替换为 `SVDLinear`，并支持指定SVD分解的秩比例。
- `SVDLinear` 支持参数量化（int8），并在推理时自动解量化。

## 5. 量化程序：`quantize_ptq.py` 使用说明

该脚本用于对DiT模型进行SVD分解和PTQ（后训练量化），并保存量化后权重。

**示例用法：**

```bash
python quantize_ptq.py \
  --ckpt pretrained_models/DiT-XL-2-256x256.pt \
  --model DiT-XL/2 \
  --num-classes 1000 \
  --rank-ratio 0.95 \
  --num-bits 8 \
  --output quantized_model.pt
```

- `--ckpt`：原始模型权重路径
- `--model`：模型类型（如 DiT-XL/2）
- `--rank-ratio`：SVD分解秩比例（如0.95）
- `--num-bits`：量化比特宽度（如8）
- `--output`：保存量化后权重的路径

## 6. 批量推理生成图片：`sample_ddp.py` 使用说明

该脚本支持多卡并行采样，生成大量图片并自动保存为 `.npz` 文件，便于后续评测。

**示例用法（多卡DDP）：**

```bash
torchrun --nnodes=1 --nproc_per_node=4 sample_ddp.py \
  --model DiT-XL/2 \
  --ckpt quantized_model.pt \
  --num-fid-samples 5000 \
  --image-size 256 \
  --rank-ratio 0.95 \
  --sample-dir ./generated
```

- `--ckpt` 可指定量化后模型权重
- `--num-fid-samples` 生成图片数量
- `--sample-dir` 图片保存目录

## 7. 图片质量评估：`evaluate_metrics.py` 使用说明

本脚本基于 [torch-fidelity](https://github.com/toshas/torch-fidelity) 评测 FID/KID/sFID 等指标。

**示例用法：**

```bash
python evaluate_metrics.py \
  --real_dir /path/to/real/images \
  --fake_dir /path/to/generated/images \
  --num_images 5000
```

- `--real_dir`：真实图片目录
- `--fake_dir`：生成图片目录
- `--num_images`：参与评测的图片数量

脚本会自动输出 FID、KID、sFID 等评测结果。

---

## 参考/致谢

- 本项目基于 [facebookresearch/DiT](https://github.com/facebookresearch/DiT)。
- SVDQuant 相关实现见 `models.py`。
- 评测部分基于 [torch-fidelity](https://github.com/toshas/torch-fidelity)。

---

如需进一步定制或补充内容，请告知！
