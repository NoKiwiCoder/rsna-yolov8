"""
文件名：train_yolov8_cbam.py
用途：实验2 - 在 YOLOv8n backbone 中添加 CBAM 注意力模块
改进点：
  - CBAM 同时建模通道注意力和空间注意力
  - 仅在 backbone 的 P3/P4/P5 层后添加，不改变 head 结构
  - 使用实际 YOLOv8n 通道数，确保与预训练权重兼容
"""
import os
import torch
import torch.nn as nn
from ultralytics import YOLO


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        mid = max(channels // reduction, 1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, mid, 1, bias=False),
            nn.SiLU(),
            nn.Conv2d(mid, channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        return x * self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        combined = torch.cat([avg_out, max_out], dim=1)
        return x * self.sigmoid(self.conv(combined))


class CBAM(nn.Module):
    def __init__(self, c1, c2=None, reduction=16, kernel_size=7):
        super().__init__()
        c2 = c2 or c1
        self.ca = ChannelAttention(c1, reduction)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.ca(x)
        x = self.sa(x)
        return x


import ultralytics.nn.tasks as tasks
tasks.parse_model.__globals__['CBAM'] = CBAM


def create_yaml(path="yolov8_cbam.yaml"):
    yaml_content = """nc: 1
backbone:
  - [-1, 1, Conv, [16, 3, 2]]
  - [-1, 1, Conv, [32, 3, 2]]
  - [-1, 1, C2f, [32, True]]
  - [-1, 1, Conv, [64, 3, 2]]
  - [-1, 2, C2f, [64, True]]
  - [-1, 1, CBAM, [64]]
  - [-1, 1, Conv, [128, 3, 2]]
  - [-1, 2, C2f, [128, True]]
  - [-1, 1, CBAM, [128]]
  - [-1, 1, Conv, [256, 3, 2]]
  - [-1, 1, C2f, [256, True]]
  - [-1, 1, SPPF, [256, 5]]
  - [-1, 1, CBAM, [256]]
head:
  - [-1, 1, nn.Upsample, [None, 2, 'nearest']]
  - [[-1, 8], 1, Concat, [1]]
  - [-1, 1, C2f, [128]]
  - [-1, 1, nn.Upsample, [None, 2, 'nearest']]
  - [[-1, 5], 1, Concat, [1]]
  - [-1, 1, C2f, [64]]
  - [-1, 1, Conv, [64, 3, 2]]
  - [[-1, 15], 1, Concat, [1]]
  - [-1, 1, C2f, [128]]
  - [-1, 1, Conv, [128, 3, 2]]
  - [[-1, 12], 1, Concat, [1]]
  - [-1, 1, C2f, [256]]
  - [[18, 21, 24], 1, Detect, [nc]]
"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(yaml_content.strip())
    return path


def main():
    yaml_path = create_yaml()
    data_path = r"C:\Users\Admin\Desktop\MedicalProject\dataset\rsna_yolo"

    print("=" * 50)
    print("Running YOLOv8n + CBAM Attention")
    print("YAML:", os.path.abspath(yaml_path))
    print("Data:", os.path.abspath(data_path))
    print("=" * 50)

    model = YOLO(yaml_path).load("yolov8n.pt")

    model.train(
        data=data_path,
        epochs=80,
        imgsz=512,
        batch=4,
        device="0",
        workers=0,
        patience=20,
        project="runs_rsna",
        name="cbam",
        exist_ok=True,
        verbose=True,
    )


if __name__ == "__main__":
    main()