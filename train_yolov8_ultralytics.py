"""
要跑不同配置的试验就复制一下这个文件，改个名字（比如 train_yolov8_original.py）,这样我们后面好管理不同版本试验
"""
import os
import torch
import torch.nn as nn
from ultralytics import YOLO


# =========================================================
# 1. EMA 模块
# =========================================================
class EMA(nn.Module):
    def __init__(self, c1, c2=None):
        super().__init__()
        c = c1
        groups = max(1, c // 16)
        while c % groups != 0:
            groups -= 1
        self.conv1x1 = nn.Conv2d(c, c, 1, bias=False)
        self.conv3x3 = nn.Conv2d(c, c, 3, padding=1, bias=False, groups=groups)
        self.bn = nn.BatchNorm2d(c)
        self.act = nn.SiLU()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        identity = x
        x1 = self.conv1x1(x)
        x2 = self.conv3x3(x)
        x = self.act(self.bn(x1 + x2))
        w = self.sigmoid(self.pool(x))
        return identity * w


# =========================================================
# 2. 注册模块
# =========================================================
import ultralytics.nn.tasks as tasks
tasks.parse_model.__globals__['EMA'] = EMA


# =========================================================
# 3. Inner-CIoU —— 直接替换 loss 模块中的 bbox_iou 函数
#    (不再修改 BboxLoss.forward，从根本上解决形状不匹配)
# =========================================================
import ultralytics.utils.loss as loss_module
from ultralytics.utils.metrics import bbox_iou as _orig_bbox_iou


def _inner_ciou(box1, box2, xywh=True, eps=1e-7):
    """
    完整重写 IoU 计算，使用与 bbox_iou 完全相同的广播规则，
    因此输出形状一定与原始 bbox_iou 一致。
    Inner-CIoU = IoU * exp(-rho^2 / c^2)
    """
    # ---- 转换 xywh → xyxy（与原始 bbox_iou 完全一致）----
    if xywh:
        b1x1 = box1[..., 0] - box1[..., 2] / 2
        b1y1 = box1[..., 1] - box1[..., 3] / 2
        b1x2 = box1[..., 0] + box1[..., 2] / 2
        b1y2 = box1[..., 1] + box1[..., 3] / 2
        b2x1 = box2[..., 0] - box2[..., 2] / 2
        b2y1 = box2[..., 1] - box2[..., 3] / 2
        b2x2 = box2[..., 0] + box2[..., 2] / 2
        b2y2 = box2[..., 1] + box2[..., 3] / 2
    else:
        b1x1, b1y1, b1x2, b1y2 = box1[..., 0], box1[..., 1], box1[..., 2], box1[..., 3]
        b2x1, b2y1, b2x2, b2y2 = box2[..., 0], box2[..., 1], box2[..., 2], box2[..., 3]

    # ---- 交集面积 ----
    inter_w = (torch.min(b1x2, b2x2) - torch.max(b1x1, b2x1)).clamp(0)
    inter_h = (torch.min(b1y2, b2y2) - torch.max(b1y1, b2y1)).clamp(0)
    inter = inter_w * inter_h

    # ---- 并集面积 ----
    w1 = (b1x2 - b1x1).clamp(0)
    h1 = (b1y2 - b1y1).clamp(0)
    w2 = (b2x2 - b2x1).clamp(0)
    h2 = (b2y2 - b2y1).clamp(0)
    union = w1 * h1 + w2 * h2 - inter + eps

    iou = inter / union

    # ---- 中心点距离平方（与原始 CIoU 完全一致）----
    cx1 = (b1x1 + b1x2) / 2
    cy1 = (b1y1 + b1y2) / 2
    cx2 = (b2x1 + b2x2) / 2
    cy2 = (b2y1 + b2y2) / 2
    rho2 = (cx1 - cx2) ** 2 + (cy1 - cy2) ** 2

    # ---- 最小外接矩形对角线平方 ----
    cw = torch.max(b1x2, b2x2) - torch.min(b1x1, b2x1)
    ch = torch.max(b1y2, b2y2) - torch.min(b1y1, b2y1)
    c2 = cw ** 2 + ch ** 2 + eps

    # ---- Inner-CIoU ----
    return iou * torch.exp(-rho2 / c2)


def _patched_bbox_iou(box1, box2, xywh=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
    """
    对 bbox_iou 的 drop-in 替换：
    - 当 CIoU=True 时，使用 Inner-CIoU
    - 其他情况（GIoU/DIoU/纯IoU），原样调用原始函数
    """
    if CIoU:
        return _inner_ciou(box1, box2, xywh=xywh, eps=eps)
    return _orig_bbox_iou(box1, box2, xywh=xywh, GIoU=GIoU, DIoU=DIoU, CIoU=CIoU, eps=eps)


# ★ 关键：把替换函数注入到 loss 模块的命名空间
# 这样 BboxLoss.forward 内部调用 bbox_iou(...) 时，实际调用的是我们的版本
loss_module.bbox_iou = _patched_bbox_iou


# =========================================================
# 4. 生成 YAML（不变）
# =========================================================
def create_yaml(path="yolov8_custom.yaml"):
    yaml_content = """nc: 3

backbone:
  # from  output          module       args
  - [-1, 1, Conv, [16, 3, 2]]        #  0  -> 16
  - [-1, 1, Conv, [32, 3, 2]]        #  1  -> 32
  - [-1, 1, C2f, [32]]               #  2  -> 32
  - [-1, 1, Conv, [64, 3, 2]]        #  3  -> 64
  - [-1, 2, C2f, [64]]               #  4  -> 64
  - [-1, 1, Conv, [128, 3, 2]]       #  5  -> 128
  - [-1, 2, C2f, [128]]              #  6  -> 128
  - [-1, 1, Conv, [256, 3, 2]]       #  7  -> 256
  - [-1, 2, C2f, [256]]              #  8  -> 256
  - [-1, 1, SPPF, [256, 5]]          #  9  -> 256

head:
  - [-1, 1, nn.Upsample, [None, 2, 'nearest']]   # 10
  - [[-1, 6], 1, Concat, [1]]                     # 11
  - [-1, 1, C2f, [128]]                           # 12
  - [-1, 1, EMA, [128]]                           # 13

  - [-1, 1, nn.Upsample, [None, 2, 'nearest']]   # 14
  - [[-1, 4], 1, Concat, [1]]                     # 15
  - [-1, 1, C2f, [64]]                            # 16
  - [-1, 1, EMA, [64]]                            # 17

  - [-1, 1, Conv, [64, 3, 2]]                     # 18
  - [[-1, 13], 1, Concat, [1]]                    # 19
  - [-1, 1, C2f, [128]]                           # 20
  - [-1, 1, EMA, [128]]                           # 21

  - [-1, 1, Conv, [128, 3, 2]]                    # 22
  - [[-1, 9], 1, Concat, [1]]                     # 23
  - [-1, 1, C2f, [256]]                           # 24
  - [-1, 1, EMA, [256]]                           # 25

  - [[17, 21, 25], 1, Detect, [nc]]               # 26
"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(yaml_content.strip())
    return path


# =========================================================
# 5. 生成假数据
# =========================================================
def create_dummy_dataset():
    import numpy as np
    from PIL import Image

    base = "dataset"
    for split in ["train", "val"]:
        os.makedirs(f"{base}/images/{split}", exist_ok=True)
        os.makedirs(f"{base}/labels/{split}", exist_ok=True)
        n = 20 if split == "train" else 5
        for i in range(n):
            img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
            Image.fromarray(img).save(f"{base}/images/{split}/{i}.jpg")
            with open(f"{base}/labels/{split}/{i}.txt", "w") as f:
                f.write("0 0.5 0.5 0.3 0.3\n")

    data_yaml = f"""path: {os.path.abspath(base)}
train: images/train
val: images/val
nc: 3
names:
  0: A
  1: B
  2: C
"""
    with open(f"{base}/data.yaml", "w", encoding="utf-8") as f:
        f.write(data_yaml.strip())
    return f"{base}/data.yaml"


# =========================================================
# 6. 主函数
# =========================================================
def main():
    yaml_path = create_yaml()
    data_path = r"C:\Users\Admin\Desktop\MedicalProject\dataset\rsna_yolo"

    print("=" * 50)
    print("YAML:", os.path.abspath(yaml_path))
    print("Data:", os.path.abspath(data_path))
    print("Loss: bbox_iou patched -> Inner-CIoU")
    print("=" * 50)

    model = YOLO(yaml_path)

    model.train(
        data=data_path,
        epochs=80,
        imgsz=512,            # 8GB 显存跑 512 很轻松；想试 1024 可以，batch 降到 4
        batch=4,             # ★ imgsz=512 时，8GB 可以跑 batch=16
        device="0",           # ★ 使用 GPU
        workers=0,            # ★ GPU 模式下 workers 可以开 4
        patience=20,
        project="runs_rsna",
        name="ema_innerciou", # ★ 实验名称，建议每次实验都修改，便于区分结果
        exist_ok=True,
        verbose=True,
    )


if __name__ == "__main__":
    main()
