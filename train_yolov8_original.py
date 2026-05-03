"""
文件名：train_yolov8_original.py
用途：作为基线实验，使用官方原版 YOLOv8 (无 EMA, 无 Inner-CIoU)
"""

import os
import torch
from ultralytics import YOLO

# =========================================================
# 1. 使用官方预定义的模型（不再需要自定义 EMA 模块）
# =========================================================
# (此处无需任何自定义类，直接使用 YOLO 默认的 'yolov8n.yaml' 或官方架构)

# =========================================================
# 2. 生成标准 YAML（使用官方默认的 yaml 名称，或者复制一份干净的）
# =========================================================
# 为了最纯粹的对比，我们直接使用 Ultralytics 内置的模型配置。
# 如果你想指定特定的大小（如 v8s），可以写 'yolov8s.yaml'。
# 这里我们直接用字符串指定，避免外部 yaml 文件干扰。

def create_yaml(path="yolov8_original.yaml"):
    # 直接使用 Ultralytics 官方默认的 YOLOv8n 架构字符串
    # 这样不需要依赖外部 yaml 文件，且保证是原版
    yaml_content = """
nc: 1
scales: # model compound scaling constants, i.e. 'model=yolov8n.yaml' will call yolov8.yaml with scale 'n'
  # [depth, width, max_channels]
  n: [0.33, 0.25, 1024]
  s: [0.33, 0.50, 1024]
  m: [0.67, 0.75, 1024]
  l: [1.00, 1.00, 1024]
  x: [1.00, 1.25, 1024]

backbone:
  # [from, repeats, module, args]
  - [-1, 1, Conv, [64, 3, 2]] # 0-P1/2
  - [-1, 1, Conv, [128, 3, 2]] # 1-P2/4
  - [-1, 3, C2f, [128, True]]
  - [-1, 1, Conv, [256, 3, 2]] # 3-P3/8
  - [-1, 6, C2f, [256, True]]
  - [-1, 1, Conv, [512, 3, 2]] # 5-P4/16
  - [-1, 6, C2f, [512, True]]
  - [-1, 1, Conv, [1024, 3, 2]] # 7-P5/32
  - [-1, 3, C2f, [1024, True]]
  - [-1, 1, SPPF, [1024, 5]] # 9

head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 6], 1, Concat, [1]] # cat backbone P4
  - [-1, 3, C2f, [512]] # 12
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 4], 1, Concat, [1]] # cat backbone P3
  - [-1, 3, C2f, [256]] # 15 (P3/8-small)
  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 12], 1, Concat, [1]] # cat head P4
  - [-1, 3, C2f, [512]] # 18 (P4/16-medium)
  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 9], 1, Concat, [1]] # cat head P5
  - [-1, 3, C2f, [1024]] # 21 (P5/32-large)
  - [[15, 18, 21], 1, Detect, [nc]] # Detect(P3, P4, P5)
"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(yaml_content.strip())
    return path

# =========================================================
# 3. 主函数（保持超参数完全一致）
# =========================================================
def main():
    # 生成原版 yaml 配置文件
    yaml_path = create_yaml()
    
    # 你的数据集路径（保持与改进版完全一致）
    data_path = r"C:\Users\Admin\Desktop\MedicalProject\dataset\rsna_yolo"
    
    print("=" * 50)
    print("Running Original YOLOv8")
    print("YAML:", os.path.abspath(yaml_path))
    print("Data:", os.path.abspath(data_path))
    print("Loss: Default CIoU (Official)")
    print("=" * 50)

    # 1. 初始化模型：直接加载官方预训练权重（等效于 YOLO("yolov8n.pt")）
    model = YOLO("yolov8n.pt") 
    
    # 2. 训练模型
    # 关键：所有参数（epochs, imgsz, batch, patience）必须与改进版完全一致
    model.train(
        data=data_path,
        epochs=80,       # 保持一致
        imgsz=512,       # 保持一致
        batch=16,         # 保持一致 (根据显存调整，但要和改进版一样)
        device="0",      # 使用 GPU 0
        workers=0,       # Windows 下有时设为 0 避免报错，或设为 4
        patience=20,     # 早停 patience
        project="runs_rsna", 
        name="original_baseline", # 实验名称改为 original_baseline，方便区分
        exist_ok=True,
        verbose=True,
    )

if __name__ == "__main__":
    main()