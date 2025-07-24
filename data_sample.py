import os
import random
import shutil
print("请输入要抽取的图片数量:")
n =int(input())
print(f"开始抽取{n}张图片...")
# 源文件夹和目标文件夹路径
src_dir = os.path.join(os.path.dirname(__file__), '/root/autodl-tmp/imagenet-1k/generated/DiT-XL-2-DiT-XL-2-256x256-size-256-vae-mse-cfg-1.5-seed-0') #val_images_5000_resized // /root/autodl-tmp/imagenet-1k/generated/DiT-XL-2-DiT-XL-2-256x256-size-256-vae-mse-cfg-1.5-seed-0
dst_dir = os.path.join(os.path.dirname(__file__), '/root/autodl-tmp/imagenet-1k/generated/DiT-XL-2-quantized_model-size-256-vae-mse-cfg-1.5-seed-0')

# 如果目标文件夹不存在，则创建
os.makedirs(dst_dir, exist_ok=True)

# 获取所有图片文件名
all_images = [f for f in os.listdir(src_dir) if f.lower().endswith('.png')]

# 随机抽取n张图片
sample_images = random.sample(all_images, n)

# 复制图片到新文件夹
for img_name in sample_images:
    src_path = os.path.join(src_dir, img_name)
    dst_path = os.path.join(dst_dir, img_name)
    shutil.copy2(src_path, dst_path)

print(f"已随机抽取{n}张图片并复制到 {dst_dir}")
