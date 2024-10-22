import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from tqdm import tqdm

# 定义数据集路径
images_tensor_path = './tensors/images_training_tensors.pt'
labels_tensor_path = './tensors/annotations_training_tensors.pt'

# 加载数据
images_tensor = torch.load(images_tensor_path)  # 输入数据
labels_tensor = torch.load(labels_tensor_path)  # 目标标签

# 创建数据集和数据加载器，并增加通道维度
images_tensor = images_tensor.unsqueeze(1)  # 将 (N, H, W) 变为 (N, 1, H, W)
labels_tensor = labels_tensor.unsqueeze(1)  # 对标签也进行相同的处理

# 创建数据集和数据加载器
dataset = TensorDataset(images_tensor, labels_tensor)
train_loader = DataLoader(dataset, batch_size=8, shuffle=True)


# attention_UNet模型定义
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




# IoU计算函数
def calculate_iou(pred: torch.Tensor, target: torch.Tensor) -> float:
    ious = []
    pred = torch.argmax(pred, dim=1)
    unique_labels = torch.unique(target)

    for cls in unique_labels:
        if cls == 0:  # 跳过背景类
            continue

        pred_class = (pred == cls)
        target_class = (target == cls)

        intersection = torch.sum((pred_class & target_class).float())
        union = torch.sum((pred_class | target_class).float())

        if union == 0:
            iou = torch.tensor(float('nan'))
        else:
            iou = intersection / union
        ious.append(iou)

    return torch.nanmean(torch.tensor(ious))


# 主程序
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNetWithAttention(in_channels=1, num_classes=4).to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    num_epochs = 50
    tolerance = 1e-5  # 收敛阈值
    previous_loss = float('inf')  # 初始化前一轮损失

    for epoch in range(num_epochs):
        model.train()  # 设置模型为训练模式
        epoch_loss = 0.0

        # 进度条
        with tqdm(total=len(train_loader), desc=f'Epoch {epoch + 1}/{num_epochs}', unit='batch') as pbar:
            for images, labels in train_loader:


                images, labels = images.to(device), labels.to(device)
                # 强制移除第三维，无论其大小如何
                images = images.squeeze(2) if images.size(2) == 1 else images[:, :, 0, :, :]
                optimizer.zero_grad()  # 清零梯度
                outputs = model(images)  # 前向传播

                # 去掉 labels 中的多余通道维度，使其从 [N, 1, H, W] 变成 [N, H, W]
                labels = labels.squeeze(1)


                loss = criterion(outputs, labels)  # 计算损失
                loss.backward()  # 反向传播
                optimizer.step()  # 更新参数

                epoch_loss += loss.item()
                pbar.set_postfix(loss=loss.item())  # 更新进度条信息
                pbar.update(1)  # 更新进度条

        epoch_loss /= len(train_loader)  # 平均损失
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}')

        # 检查损失收敛
        if abs(previous_loss - epoch_loss) < tolerance:
            print(f"Loss has converged at epoch {epoch + 1}")
            break
        previous_loss = epoch_loss  # 更新前一轮损失

    # 保存整个模型
    torch.save(model, './models/attention_UNet_model.pth')
    print("Model saved to 'attention_UNet_model.pth'")

    print("Training complete.")
