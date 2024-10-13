import os
import torch
import numpy as np
from PIL import Image

# 定义图像文件夹路径
folder_path = './data/annotations/test'  # 替换为你的文件夹路径

# 获取文件夹中的所有图片文件名（.png格式）
image_files = [f for f in os.listdir(folder_path) if f.endswith('.png')]

# 用于存储所有图片的张量列表
tensor_list = []

# 遍历每张图片，加载并转换为Tensor
for image_file in image_files:
    img_path = os.path.join(folder_path, image_file)

    # 加载灰度图像
    img = Image.open(img_path).convert('L')

    # 将图像转为 NumPy 数组，并保持原来的 [0, 255] 值
    img_np = np.array(img)

    # 将 NumPy 数组转换为 PyTorch 张量，并保持为整数类型
    img_tensor = torch.from_numpy(img_np).long()

    # 不增加通道维度，保持 (H, W) 形式
    tensor_list.append(img_tensor)

    # 打印转换后的张量形状和文件名
    print(f'Processing {image_file} - Tensor shape: {img_tensor.shape}')

# 将所有张量堆叠成一个大张量，形状为 (N, H, W)
all_tensors = torch.stack(tensor_list)

# 定义保存所有张量的文件路径
tensor_file_path = './tensors/annotations_test_tensors.pt'

# 保存合并后的张量到一个文件
torch.save(all_tensors, tensor_file_path)
print(f'All tensors saved to {tensor_file_path}')

# 读取保存的张量
all_tensors = torch.load(tensor_file_path)

# 输出张量的形状
print(f'Loaded Tensor shape: {all_tensors.shape}')
