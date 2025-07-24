# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A script to perform Post-Training Quantization (PTQ) on a DiT model with SVDQuant.
"""
import torch
import argparse
import os
from models import DiT_models, apply_svd_to_dit, SVDLinear


def quantize_all_svdlinear(model, num_bits=8):
    """
    遍历模型，将所有SVDLinear层做PTQ量化。
    """
    for m in model.modules():
        if isinstance(m, SVDLinear):
            m.quantize_parameters(num_bits=num_bits)


def main(args):
    assert os.path.exists(args.ckpt), f"Checkpoint {args.ckpt} not found!"
    # 1. 加载模型
    model = DiT_models[args.model](
        input_size=32,
        num_classes=args.num_classes,
        learn_sigma=True,
    )
    state_dict = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(state_dict)
    print("Pre-trained weights loaded.")

    # 自动转移到GPU（如可用）
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # 2. SVD分解
    model = apply_svd_to_dit(model, rank_ratio=args.rank_ratio)
    print("SVD decomposition applied.")

    # # 3. PTQ权重量化
    # quantize_all_svdlinear(model, num_bits=args.num_bits)
    # print(f"PTQ quantization (int{args.num_bits}) applied to all SVDLinear layers.")

    # 4. 保存量化后模型（转回cpu保存）
    model = model.cpu()
    torch.save(model.state_dict(), args.output)
    print(f"Quantized model saved to {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="pretrained_models/DiT-XL-2-256x256.pt", help="Path to the pre-trained DiT checkpoint. 默认: pretrained_models/DiT-XL-2-256x256.pt")
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--rank-ratio", type=float, default=0.95, help="SVD rank ratio for quantization.")
    parser.add_argument("--num-bits", type=int, default=8, help="Quantization bit width (default: 8)")
    parser.add_argument("--output", type=str, default="quantized_model.pt", help="Output path for quantized model.")
    args = parser.parse_args()
    main(args) 