import torch
import time
import os
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import torch.nn as nn
# UNet模型定义（与训练时相同）
class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 全局平均池化
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
class DoubleConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        if mid_channels is None:
            mid_channels = out_channels
        super(DoubleConv, self).__init__(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

class Down(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__(
            nn.MaxPool2d(2, stride=2),
            DoubleConv(in_channels, out_channels)
        )

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x

class OutConv(nn.Sequential):
    def __init__(self, in_channels, num_classes):
        super(OutConv, self).__init__(
            nn.Conv2d(in_channels, num_classes, kernel_size=1)
        )

class UNet(nn.Module):
    def __init__(self, in_channels: int = 1, num_classes: int = 4, bilinear: bool = True, base_c: int = 64):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.bilinear = bilinear

        self.in_conv = DoubleConv(in_channels, base_c)
        self.down1 = Down(base_c, base_c * 2)
        self.down2 = Down(base_c * 2, base_c * 4)
        self.down3 = Down(base_c * 4, base_c * 8)
        factor = 2 if bilinear else 1
        self.down4 = Down(base_c * 8, base_c * 16 // factor)
        self.up1 = Up(base_c * 16, base_c * 8 // factor, bilinear)
        self.up2 = Up(base_c * 8, base_c * 4 // factor, bilinear)
        self.up3 = Up(base_c * 4, base_c * 2 // factor, bilinear)
        self.up4 = Up(base_c * 2, base_c, bilinear)
        self.out_conv = OutConv(base_c, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.in_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.out_conv(x)
        return logits

class UNetWithAttention(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 3, bilinear: bool = True, base_c: int = 64):
        super(UNetWithAttention, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.bilinear = bilinear

        # 下采样部分
        self.in_conv = DoubleConv(in_channels, base_c)
        self.se1 = SEBlock(base_c)  # 注意力模块
        self.down1 = Down(base_c, base_c * 2)
        self.se2 = SEBlock(base_c * 2)  # 注意力模块
        self.down2 = Down(base_c * 2, base_c * 4)
        self.se3 = SEBlock(base_c * 4)  # 注意力模块
        self.down3 = Down(base_c * 4, base_c * 8)
        factor = 2 if bilinear else 1
        self.down4 = Down(base_c * 8, base_c * 16 // factor)
        self.se4 = SEBlock(base_c * 16 // factor)  # 注意力模块

        # 上采样部分
        self.up1 = Up(base_c * 16, base_c * 8 // factor, bilinear)
        self.se5 = SEBlock(base_c * 8 // factor)  # 注意力模块
        self.up2 = Up(base_c * 8, base_c * 4 // factor, bilinear)
        self.se6 = SEBlock(base_c * 4 // factor)  # 注意力模块
        self.up3 = Up(base_c * 4, base_c * 2 // factor, bilinear)
        self.se7 = SEBlock(base_c * 2 // factor)  # 注意力模块
        self.up4 = Up(base_c * 2, base_c, bilinear)
        self.se8 = SEBlock(base_c)  # 注意力模块

        # 输出层
        self.out_conv = OutConv(base_c, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.se1(self.in_conv(x))
        x2 = self.se2(self.down1(x1))
        x3 = self.se3(self.down2(x2))
        x4 = self.se4(self.down3(x3))
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.se5(x)
        x = self.up2(x, x3)
        x = self.se6(x)
        x = self.up3(x, x2)
        x = self.se7(x)
        x = self.up4(x, x1)
        x = self.se8(x)
        logits = self.out_conv(x)
        return logits


# 加载模型
model = torch.load('../model/UNet_model.pth')
model.eval()

# 定义测试集张量
test_images_tensor = torch.load('../tensors/images_test_tensors.pt')
test_labels_tensor = torch.load('../tensors/annotations_test_tensors.pt')

test_images_tensor = test_images_tensor.unsqueeze(1)
test_labels_tensor = test_labels_tensor.unsqueeze(1)

# 创建测试集 DataLoader
test_dataset = TensorDataset(test_images_tensor, test_labels_tensor)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# 确保保存结果的文件夹存在
os.makedirs('baseline_predictions', exist_ok=True)
os.makedirs('test_predictions', exist_ok=True)
os.makedirs('test_ground_truths', exist_ok=True)

# 测试并保存分割结果
with torch.no_grad():
    with tqdm(total=len(test_loader), desc="Testing", unit="batch") as pbar:
        for i, (images, gt_labels) in enumerate(test_loader):
            images, gt_labels = images.to('cuda'), gt_labels.to('cuda')  # 移动到GPU

            outputs = model(images)
            outputs = F.softmax(outputs, dim=1)
            predictions = torch.argmax(outputs, dim=1)

            # 获取当前测试图像的文件名
            file_name = f"{i + 1:06d}.jpg"  # 根据你的测试文件命名规则

            # 保存分割结果
            np.save(f'baseline_predictions/prediction_{file_name[:-4]}.npy', predictions.cpu().numpy())
            np.save(f'test_ground_truths/ground_truth_{file_name[:-4]}.npy', gt_labels.squeeze(1).cpu().numpy())

            # 假设你还有其他模型的分割结果（替换为你其他模型的前向传播代码）
            unet_model = torch.load('../model/attention_UNet_model.pth')
            unet_model.eval()
            other_model_outputs = unet_model(images)

            other_predictions = torch.argmax(F.softmax(other_model_outputs, dim=1), dim=1)
            np.save(f'test_predictions/prediction_{file_name[:-4]}.npy', other_predictions.cpu().numpy())

            pbar.update(1)

print("测试完成，分割结果已保存。")
