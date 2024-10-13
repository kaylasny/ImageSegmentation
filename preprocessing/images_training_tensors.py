import os
import torch
from torchvision import transforms
from PIL import Image

# 定义图像文件夹路径
folder_path = './data/images/training'  # 替换为你的文件夹路径

# 定义转化为Tensor的变换
transform = transforms.ToTensor()

# 获取文件夹中的所有图片文件名（.png格式）
image_files = [f for f in os.listdir(folder_path) if f.endswith('.jpg')]

# 用于存储所有图片的张量列表
tensor_list = []

# 遍历每张图片，加载并转换为Tensor
for image_file in image_files:
    img_path = os.path.join(folder_path, image_file)

    # 加载灰度图像
    img = Image.open(img_path).convert('L')

    # 将图像转化为Tensor
    img_tensor = transform(img)

    # 去掉通道维度，确保是 (H, W)
    img_tensor = img_tensor.squeeze(0)

    # 将每个图像的张量添加到列表中
    tensor_list.append(img_tensor)

    # 打印转换后的张量形状和文件名
    print(f'Processing {image_file} - Tensor shape: {img_tensor.shape}')

# 将所有张量堆叠成一个大张量，形状为 (N, H, W)
all_tensors = torch.stack(tensor_list)

# 定义保存所有张量的文件路径
tensor_file_path = './tensors/images_training_tensors.pt'

# 保存合并后的张量到一个文件
torch.save(all_tensors, tensor_file_path)
print(f'All tensors saved to {tensor_file_path}')

# 读取保存的张量
all_tensors = torch.load(tensor_file_path)

# 输出张量的形状
print(f'Loaded Tensor shape: {all_tensors.shape}')
