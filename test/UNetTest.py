import torch
import time
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from tqdm import tqdm
import torch.nn as nn
import math

# UNet模型定义
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
    def __init__(self, in_channels: int = 3, num_classes: int = 3, bilinear: bool = True, base_c: int = 64):
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

# 加载模型
model = torch.load('./models/UNet_model.pth')
model.eval()

# 定义测试集张量
test_images_tensor = torch.load('./tensors/images_test_tensors.pt')
test_labels_tensor = torch.load('./tensors/annotations_test_tensors.pt')

test_images_tensor = test_images_tensor.unsqueeze(1)
test_labels_tensor = test_labels_tensor.unsqueeze(1)

# 创建测试集 DataLoader
test_dataset = TensorDataset(test_images_tensor, test_labels_tensor)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# IoU 计算函数
def calculate_iou(pred, target):
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

        iou = intersection / union if union != 0 else torch.tensor(float('nan'))
        ious.append(iou)

    return torch.nanmean(torch.tensor(ious)), ious

# 测试并计算 IoU 和 FPS
total_iou = 0
total_images = 0
total_time = 0
ious_list = []

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# 为测试过程添加进度条
with torch.no_grad():
    with tqdm(total=len(test_loader), desc="Testing", unit="batch") as pbar:
        for images, gt_labels in test_loader:
            images, gt_labels = images.to(device), gt_labels.to(device)

            start_time = time.time()
            outputs = model(images)
            outputs = F.softmax(outputs, dim=1)

            end_time = time.time()
            total_time += end_time - start_time

            avg_iou, ious_per_class = calculate_iou(outputs, gt_labels.squeeze(1))
            total_iou += avg_iou.item()
            ious_list.append(ious_per_class)
            total_images += 1

            pbar.set_postfix(avg_iou=avg_iou.item())
            pbar.update(1)

# 平均 IoU 和 FPS 计算
mean_iou = total_iou / total_images
fps = total_images / total_time

# 计算模型参数量
num_params = sum(p.numel() for p in model.parameters())

# 输出结果
print(f"平均交并比（Mean IoU）: {mean_iou:.4f}")
print(f"每秒帧数（FPS）: {fps:.2f}")
print(f"模型参数量: {num_params / 1e6:.2f}MB")  # 转换为 MB

# 定义类别名称
class_names = {
    1: "缺陷类型 1",
    2: "缺陷类型 2",
    3: "缺陷类型 3"
}

# 定义每个缺陷类别的 IoU 存储
class_iou_sum = {1: 0.0, 2: 0.0, 3: 0.0}
class_iou_count = {1: 0, 2: 0, 3: 0}

for i, ious_per_image in enumerate(ious_list):
    ious_per_image = [iou.item() for iou in ious_per_image]  # 转换为浮点数
    output_str = f"测试图片 {i + 1} 的各类缺陷交并比: "
    output_str += ", ".join(f"{class_names[j]}: {ious_per_image[j-1]:.2f}" for j in range(1, len(ious_per_image) + 1))
    print(output_str)

    # 累加每类缺陷的 IoU
    for j in range(1, len(ious_per_image) + 1):
        if not math.isnan(ious_per_image[j-1]):
            class_iou_sum[j] += ious_per_image[j-1]
            class_iou_count[j] += 1

# 计算每类缺陷的平均 IoU
average_iou = {cls: (class_iou_sum[cls] / class_iou_count[cls] if class_iou_count[cls] > 0 else float('nan')) for cls in class_iou_sum}

# 输出每类缺陷的平均 IoU
for cls, avg in average_iou.items():
    print(f"{class_names[cls]} 的平均 IoU: {avg:.2f}")
