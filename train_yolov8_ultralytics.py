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
    # ---- 转换 xywh → xyxy ----
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
    # ---- 中心点距离与外接矩形 ----
    cx1 = (b1x1 + b1x2) / 2
    cy1 = (b1y1 + b1y2) / 2
    cx2 = (b2x1 + b2x2) / 2
    cy2 = (b2y1 + b2y2) / 2
    rho2 = (cx1 - cx2) ** 2 + (cy1 - cy2) ** 2
    cw = torch.max(b1x2, b2x2) - torch.min(b1x1, b2x1)
    ch = torch.max(b1y2, b2y2) - torch.min(b1y1, b2y1)
    c2 = cw ** 2 + ch ** 2 + eps
    # ★ 必须补充宽高比惩罚项，否则退化严重
    import math
    v = (4 / math.pi ** 2) * (torch.atan(w2 / (h2 + eps)) - torch.atan(w1 / (h1 + eps))) ** 2
    with torch.no_grad():
        alpha = v / (1 - iou + v + eps)
    # ---- Inner-CIoU ----
    # 采用乘法衰减的同时，减去宽高比惩罚
    return iou * torch.exp(-rho2 / c2) - v * alpha


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
    - [-1, 1, C2f, [32, True]]         #  2  -> 32 (去掉多余的1，True代表shortcut)
    - [-1, 1, Conv, [64, 3, 2]]        #  3  -> 64
    - [-1, 2, C2f, [64, True]]         #  4  -> 64
    - [-1, 1, Conv, [128, 3, 2]]       #  5  -> 128
    - [-1, 2, C2f, [128, True]]        #  6  -> 128
    - [-1, 1, Conv, [256, 3, 2]]       #  7  -> 256
    - [-1, 1, C2f, [256, True]]        #  8  -> 256
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
  - [[17, 21, 25], 1, Detect, [nc]]               # 26 (严格对齐原版3头输出)
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

    # ★ 关键修改：从 YAML 构建模型，并强行加载 yolov8n.pt 的预训练权重
    model = YOLO(yaml_path).load("yolov8n.pt")

    model.train(
        data=data_path,
        epochs=80,
        imgsz=1024,           # 扩展为 1024
        batch=4,              # 1024分辨率，batch降为4
        device="0",           
        workers=0,            # Colab环境可以开4
        patience=20,
        project="runs_rsna",
        name="ema_innerciou_1024_pretrained", # 修改实验名
        exist_ok=True,
        verbose=True,
        # 建议补充以下参数，与原版绝对对齐
        amp=True,
        box=7.5,
        cls=0.5,
        dfl=1.5
    )
    # # 指向中断时保存的 last.pt 权重文件路径
    # last_pt_path = r"runs\detect\runs_rsna\ema_innerciou_1024_pretrained\weights\last.pt"
    
    # # 直接加载 last.pt，无需再指定 yaml 和原预训练权重
    # model = YOLO(last_pt_path)
    
    # # 开启 resume=True，框架会自动继承中断时的 epoch、优化器状态、学习率等
    # model.train(resume=True)


if __name__ == "__main__":
    main()
