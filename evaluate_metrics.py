import os
import time
import subprocess
import argparse
import sys
from torch_fidelity import calculate_metrics

def evaluate_fid(real_dir, fake_dir):
    """
    使用 torch-fidelity 评测 FID, KID 和 sFID（直接用Python API）
    """
    print("[INFO] 开始评测 FID, KID 和 sFID ...")
    try:
        metrics = calculate_metrics(
            input1=real_dir,
            input2=fake_dir,
            cuda=True,  
            isc=True,
            fid=True,
            kid=True,
            sfid=True,
            verbose=True
        )
        print("[RESULT] 评测结果:")
        for k, v in metrics.items():
            print(f"  {k}: {v}")
    except Exception as e:
        print(f"[ERROR] FID/sFID 评测失败: {e}")
        exit(1)

def evaluate_clip_score(fake_dir, captions_file):
  
    executable_path = os.path.join(os.path.dirname(sys.executable), 'clip_score')
    cmd = [
        executable_path,
        '--images', fake_dir,
        '--captions', captions_file
    ]
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError:
        print(f"[ERROR] 'clip_score' 可执行文件在路径 '{executable_path}' 未找到。")
        print("[INFO] 请先安装 clip-retrieval: pip install clip-retrieval")
        exit(1)
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] CLIP Score 评测失败: {e}")
        exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DiT 模型生成图片质量和性能评估脚本")
    parser.add_argument('--real_dir', type=str, required=True, help='真实图片所在目录路径.')
    parser.add_argument('--fake_dir', type=str, required=True, help='模型生成图片所在目录路径.')
    parser.add_argument('--num_images', type=int, default=5000, help='目录中预期的图片数量.')
    # parser.add_argument('--captions_file', type=str, default=None, help='CLIP Score 所需的描述文件路径. 如果不提供，则跳过 CLIP Score 评测.')
    # 使用 parse_known_args 来分离主脚本和子脚本的参数
    args, sample_args = parser.parse_known_args()

    # --- 评测FID/sFID ---
    print("-" * 50)
    # 检查路径是否存在
    if not os.path.isdir(args.real_dir):
        print(f"[ERROR] 真实图片目录不存在: {args.real_dir}")
        exit(1)
    if not os.path.isdir(args.fake_dir):
        print(f"[ERROR] 生成图片目录不存在: {args.fake_dir}")
        exit(1)
    # 检查图片数量
    real_images_count = len([name for name in os.listdir(args.real_dir) if os.path.isfile(os.path.join(args.real_dir, name))])
    fake_images_count = len([name for name in os.listdir(args.fake_dir) if os.path.isfile(os.path.join(args.fake_dir, name))])
    if real_images_count < args.num_images:
        print(f"[WARN] 真实图片目录 {args.real_dir} 中只有 {real_images_count} 张图片, 少于预期的 {args.num_images} 张.")
    if fake_images_count < args.num_images:
        print(f"[WARN] 生成图片目录 {args.fake_dir} 中只有 {fake_images_count} 张图片, 少于预期的 {args.num_images} 张.")
    evaluate_fid(args.real_dir, args.fake_dir)

    # # --- 2. 评测CLIP Score ---
    # print("-" * 50)
    # if args.captions_file:
    #     if os.path.exists(args.captions_file):
    #         evaluate_clip_score(args.fake_dir, args.captions_file)
    #     else:
    #         print(f"[WARN] 未找到描述文件 {args.captions_file}，跳过CLIP Score评测。")
    # else:
    #     print("[INFO] 未提供描述文件 (--captions_file)，跳过CLIP Score评测。")



    print("-" * 50)
    print("[INFO] 所有评测完成！") 