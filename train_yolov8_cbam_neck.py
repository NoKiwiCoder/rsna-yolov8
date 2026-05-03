"""
文件名：train_yolov8_cbam_neck.py
用途：实验 - CBAM 放在 Neck 的 C2f 之后（特征融合后加注意力）
优势：
  - Backbone 完全不变，预训练权重 100% 直接加载，无需手动映射
  - CBAM 作用于融合后特征，对跨尺度检测更有意义
  - 肺炎病灶跨尺度变化大，Neck 注意力帮助筛选融合信息
"""
import os
import torch
import torch.nn as nn
from ultralytics import YOLO


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        mid = max(channels // reduction, 1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, mid, 1, bias=False),
            nn.SiLU(),
            nn.Conv2d(mid, channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(x.amax(dim=(2, 3), keepdim=True))
        return self.sigmoid(avg_out + max_out)


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
        return self.sigmoid(self.conv(combined))


class CBAM(nn.Module):
    def __init__(self, c1, c2=None, reduction=8, kernel_size=7):
        super().__init__()
        c2 = c2 or c1
        self.ca = ChannelAttention(c1, reduction)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        ca_w = self.ca(x)
        sa_w = self.sa(x * ca_w)
        return x + x * ca_w * sa_w


import ultralytics.nn.tasks as tasks
tasks.parse_model.__globals__['CBAM'] = CBAM


CBAM_INSERT_AFTER = [12, 15]


def load_pretrained_with_mapping(model, pretrained_path="yolov8n.pt"):
    pretrained = YOLO(pretrained_path)
    pretrained_sd = pretrained.model.state_dict()
    cbam_sd = model.model.state_dict()

    mapped = {}
    matched = 0
    for key, value in pretrained_sd.items():
        parts = key.split('.')
        layer_idx = int(parts[1])

        offset = sum(1 for i in CBAM_INSERT_AFTER if i < layer_idx)
        new_idx = layer_idx + offset

        new_key = 'model.' + '.'.join([str(new_idx)] + parts[2:])

        if new_key in cbam_sd and cbam_sd[new_key].shape == value.shape:
            mapped[new_key] = value
            matched += 1

    cbam_sd.update(mapped)
    model.model.load_state_dict(cbam_sd)

    total = len(pretrained_sd)
    print(f"Pretrained weights: {matched}/{total} parameters loaded")
    print(f"CBAM inserted after layers {CBAM_INSERT_AFTER}")

    return model


def create_yaml(path="yolov8_cbam_neck.yaml"):
    yaml_content = """nc: 1
backbone:
  - [-1, 1, Conv, [16, 3, 2]]
  - [-1, 1, Conv, [32, 3, 2]]
  - [-1, 1, C2f, [32, True]]
  - [-1, 1, Conv, [64, 3, 2]]
  - [-1, 2, C2f, [64, True]]
  - [-1, 1, Conv, [128, 3, 2]]
  - [-1, 2, C2f, [128, True]]
  - [-1, 1, Conv, [256, 3, 2]]
  - [-1, 1, C2f, [256, True]]
  - [-1, 1, SPPF, [256, 5]]
head:
  - [-1, 1, nn.Upsample, [None, 2, 'nearest']]
  - [[-1, 6], 1, Concat, [1]]
  - [-1, 1, C2f, [128]]
  - [-1, 1, CBAM, [128]]
  - [-1, 1, nn.Upsample, [None, 2, 'nearest']]
  - [[-1, 4], 1, Concat, [1]]
  - [-1, 1, C2f, [64]]
  - [-1, 1, CBAM, [64]]
  - [-1, 1, Conv, [64, 3, 2]]
  - [[-1, 13], 1, Concat, [1]]
  - [-1, 1, C2f, [128]]
  - [-1, 1, Conv, [128, 3, 2]]
  - [[-1, 9], 1, Concat, [1]]
  - [-1, 1, C2f, [256]]
  - [[17, 20, 23], 1, Detect, [nc]]
"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(yaml_content.strip())
    return path


def main():
    yaml_path = create_yaml()
    data_path = r"C:\Users\Admin\Desktop\MedicalProject\dataset\rsna_yolo"

    print("=" * 50)
    print("Running YOLOv8n + CBAM in Neck (after C2f in FPN/PAN)")
    print("YAML:", os.path.abspath(yaml_path))
    print("Data:", os.path.abspath(data_path))
    print("=" * 50)

    model = YOLO(yaml_path)
    model = load_pretrained_with_mapping(model, "yolov8n.pt")

    model.train(
        data=data_path,
        epochs=80,
        imgsz=512,
        batch=16,
        device="0",
        workers=0,
        patience=20,
        project="runs_rsna",
        name="cbam_neck",
        exist_ok=True,
        verbose=True,
    )


if __name__ == "__main__":
    main()
