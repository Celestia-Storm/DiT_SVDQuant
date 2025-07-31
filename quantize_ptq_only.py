# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A script to perform Post-Training Quantization (PTQ) on a DiT model WITHOUT SVD.
This is to test if quantization alone can reduce model size.
"""
import torch
import argparse
import os
from models import DiT_models


def quantize_linear_layer(layer, num_bits=8):
    """
    对单个nn.Linear层进行量化
    """
    def quantize_tensor(tensor, num_bits=8):
        qmin = -2**(num_bits-1)
        qmax = 2**(num_bits-1) - 1
        scale = tensor.abs().max() / qmax if tensor.abs().max() > 0 else 1.0
        q_tensor = (tensor / scale).round().clamp(qmin, qmax).to(torch.int8)
        return q_tensor, scale

    # 量化权重
    q_weight, scale_weight = quantize_tensor(layer.weight.data, num_bits)
    layer.register_buffer('q_weight', q_weight)
    layer.register_buffer('scale_weight', torch.tensor(scale_weight))
    
    # 量化偏置（如果有）
    if layer.bias is not None:
        q_bias, scale_bias = quantize_tensor(layer.bias.data, num_bits)
        layer.register_buffer('q_bias', q_bias)
        layer.register_buffer('scale_bias', torch.tensor(scale_bias))
    else:
        layer.register_buffer('q_bias', torch.zeros(1, dtype=torch.int8))
        layer.register_buffer('scale_bias', torch.tensor(1.0))
    
    layer.is_quantized = True


def dequantize_linear_layer(layer, x):
    """
    反量化并执行前向传播
    """
    if not hasattr(layer, 'is_quantized') or not layer.is_quantized:
        return layer(x)
    
    # 反量化权重
    weight = layer.q_weight.float() * layer.scale_weight
    
    # 反量化偏置
    if layer.bias is not None:
        bias = layer.q_bias.float() * layer.scale_bias
    else:
        bias = None
    
    # 执行线性变换
    return torch.nn.functional.linear(x, weight, bias)


def quantize_all_linear(model, num_bits=8):
    """
    遍历模型，将所有nn.Linear层做PTQ量化
    """
    quantized_count = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            quantize_linear_layer(module, num_bits)
            quantized_count += 1
            print(f"Quantized layer: {name}")
    print(f"Total quantized layers: {quantized_count}")


def count_parameters(model):
    """计算模型的参数数量"""
    total_params = 0
    for name, param in model.named_parameters():
        total_params += param.numel()
    return total_params


def count_buffers(model):
    """计算模型的buffer数量"""
    total_buffers = 0
    for name, buffer in model.named_buffers():
        total_buffers += buffer.numel()
    return total_buffers


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
    
    # 计算原始模型参数
    original_params = count_parameters(model)
    original_buffers = count_buffers(model)
    print(f"Original model - Parameters: {original_params:,}, Buffers: {original_buffers:,}")

    # 2. PTQ权重量化（不做SVD）
    quantize_all_linear(model, num_bits=args.num_bits)
    print(f"PTQ quantization (int{args.num_bits}) applied to all Linear layers.")
    
    # 计算量化后的参数
    quantized_params = count_parameters(model)
    quantized_buffers = count_buffers(model)
    print(f"After quantization - Parameters: {quantized_params:,}, Buffers: {quantized_buffers:,}")
    
    print(f"Parameter change: {(quantized_params-original_params)/original_params*100:.1f}%")
    print(f"Buffer change: {(quantized_buffers-original_buffers)/original_buffers*100:.1f}%" if original_buffers > 0 else "No original buffers")

    # 3. 保存量化后模型
    torch.save(model.state_dict(), args.output)
    print(f"Quantized model saved to {args.output}")


if __name__ == "__main__":
    import torch.nn as nn
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="pretrained_models/DiT-XL-2-256x256.pt", help="Path to the pre-trained DiT checkpoint.")
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--num-bits", type=int, default=8, help="Quantization bit width (default: 8)")
    parser.add_argument("--output", type=str, default="quantized_model_ptq_only.pt", help="Output path for quantized model.")
    args = parser.parse_args()
    main(args) 