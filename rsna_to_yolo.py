"""
将 RSNA Pneumonia Detection Challenge 的 DICOM + CSV 转换为 Ultralytics YOLOv8 格式。
- 输入：Kaggle 原始 stage_2_train_labels.csv + stage_2_train_images/*.dcm
- 输出：
    images/train/*.jpg
    images/val/*.jpg
    labels/train/*.txt  （YOLO 归一化：0 x_center y_center width height）
    labels/val/*.txt
    data.yaml
"""
import os
import csv
import argparse
import random
import math
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut  # 可用 VOI LUT

# ============ 可调参数 ============
INPUT_ROOT = r"C:\Users\Admin\Desktop\MedicalProject\dataset\rsna-pneumonia-detection-challenge"  # 改成你解压后的路径
OUTPUT_ROOT = r"C:\Users\Admin\Desktop\MedicalProject\dataset\rsna_yolo"  # YOLO 输出根目录
VAL_RATIO = 0.15           # 验证集占比（建议按患者划分）
IMG_SIZE = 1024            # 输出图像边长（正方形），建议先 512/1024，显存不够就 512
USE_VOI_LUT = True         # 是否应用 DICOM 的 VOI LUT（推荐；若无则回退到简单的像素裁剪与归一化）
SEED = 42
# ================================


def maybe_apply_voi_lut(ds):
    """
    尝试使用 VOI LUT（窗宽窗位）把像素转为更好的对比度；
    若不可用则回退为简单的 0-255 归一化。
    """
    pixel_array = ds.pixel_array.astype(float)
    # 优先使用 VOI LUT
    if USE_VOI_LUT:
        voi_lut = apply_voi_lut(pixel_array, ds)
        if voi_lut is not None:
            pixel_array = voi_lut.astype(float)
    # 若仍未做过归一化，则做一个简单的 min-max 到 0-255
    if pixel_array.max() > 255 or pixel_array.min() < 0:
        mn, mx = pixel_array.min(), pixel_array.max()
        if mx - mn > 1e-6:
            pixel_array = (pixel_array - mn) / (mx - mn) * 255.0
        else:
            pixel_array = pixel_array * 0
    return pixel_array.astype("uint8")


def load_dcm_as_rgb(path):
    """读取一张 DICOM，返回 (H,W,3) 的 RGB 图像（单通道复制为三通道）"""
    ds = pydicom.dcmread(path)
    arr = maybe_apply_voi_lut(ds)
    # 单通道 -> 三通道
    img = Image.fromarray(arr, mode="L").convert("RGB")
    return img


def make_square_and_resize(img, size=1024):
    """
    将图像居中补黑边成 size×size 后 resize 为 size×size。
    你也可以改成直接 resize 或 fit 到固定尺寸，这里是最简单的一种。
    """
    w, h = img.size
    # 计算缩放因子，使得长边= size
    scale = size / max(w, h)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    img = img.resize((new_w, new_h), Image.BILINEAR)
    # 创建黑色底图并居中贴图
    canvas = Image.new("RGB", (size, size), (0, 0, 0))
    paste_x = (size - new_w) // 2
    paste_y = (size - new_h) // 2
    canvas.paste(img, (paste_x, paste_y))
    return canvas, scale, paste_x, paste_y


def build_yolo_line(x_min, y_min, bbox_w, bbox_h,
                    orig_w, orig_h,
                    scale, paste_x, paste_y):
    """
    把原始 (x_min, y_min, bbox_w, bbox_h) 转成 YOLO 归一化 (x_center, y_center, w, h)。
    注意要先对坐标做与图像相同的 resize+padding 变换。
    """
    # 左上角缩放
    x1 = x_min * scale + paste_x
    y1 = y_min * scale + paste_y
    bw = bbox_w * scale
    bh = bbox_h * scale

    # YOLO 归一化（相对 resize 后的正方形边长 size）
    x_center = (x1 + bw / 2.0) / IMG_SIZE
    y_center = (y1 + bh / 2.0) / IMG_SIZE
    w_norm = bw / IMG_SIZE
    h_norm = bh / IMG_SIZE

    # 限定在 0~1
    x_center = max(0.0, min(1.0, x_center))
    y_center = max(0.0, min(1.0, y_center))
    w_norm = max(0.0, min(1.0, w_norm))
    h_norm = max(0.0, min(1.0, h_norm))

    # 类别：0=肺炎（RSNA 是二分类检测，有框即为 1）
    return f"0 {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}"


def main():
    # 创建输出目录
    for sub in ["images/train", "images/val", "labels/train", "labels/val"]:
        os.makedirs(os.path.join(OUTPUT_ROOT, sub), exist_ok=True)

    # 读取 CSV
    csv_path = os.path.join(INPUT_ROOT, "stage_2_train_labels.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            "找不到 stage_2_train_labels.csv，请检查 INPUT_ROOT 是否正确。"
        )

    # 先按 patientId 分组（一个 patientId 可能对应多行）
    # 同时收集所有有框的 patientId（用于划分 train/val）
    patient_to_boxes = {}      # {patientId: [(x,y,w,h), ...]}
    patient_to_target = {}     # {patientId: 0 or 1}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = row["patientId"]
            target = int(row["Target"])
            patient_to_target[pid] = target
            if target == 1:
                x = float(row["x"])
                y = float(row["y"])
                w = float(row["width"])
                h = float(row["height"])
                patient_to_boxes.setdefault(pid, []).append((x, y, w, h))

    # 把“有肺炎框”的患者全部用于训练/验证划分；
    # 若想加入无肺炎样本作为负样本，可取消下面的注释（显存/时间允许的话建议加）：
    use_negatives = True   # 改成 False 可跳过大量背景样本

    all_patients = list(patient_to_boxes.keys())
    if use_negatives:
        # 把 target==0 的患者也加入（按需），负样本也按患者级别划分
        neg_patients = [pid for pid, t in patient_to_target.items() if t == 0]
        all_patients = list(set(all_patients + neg_patients))

    # 随机划分患者（保证同一患者只出现在 train 或 val）
    random.seed(SEED)
    random.shuffle(all_patients)
    n_val = max(1, int(len(all_patients) * VAL_RATIO))
    val_set = set(all_patients[:n_val])
    train_set = set(all_patients[n_val:])

    # 统计
    train_pos = len(train_set & set(patient_to_boxes.keys()))
    train_neg = len(train_set) - train_pos
    val_pos = len(val_set & set(patient_to_boxes.keys()))
    val_neg = len(val_set) - val_pos

    print(f"患者划分（val_ratio={VAL_RATIO}）:")
    print(f"  train: {len(train_set)} patients ({train_pos} positive, {train_neg} negative)")
    print(f"  val:   {len(val_set)} patients ({val_pos} positive, {val_neg} negative)")

    # 遍历处理
    dcm_dir = os.path.join(INPUT_ROOT, "stage_2_train_images")

    stats = {"train": 0, "val": 0}
    for pid in tqdm(all_patients, desc="Converting"):
        split = "val" if pid in val_set else "train"
        dcm_path = os.path.join(dcm_dir, f"{pid}.dcm")
        if not os.path.exists(dcm_path):
            continue

        try:
            img_pil = load_dcm_as_rgb(dcm_path)
        except Exception as e:
            print(f"[WARN] 读取失败，跳过: {pid}, {e}")
            continue

        orig_w, orig_h = img_pil.size
        img_out, scale, px, py = make_square_and_resize(img_pil, size=IMG_SIZE)

        # 保存图像
        out_img_path = os.path.join(OUTPUT_ROOT, "images", split, f"{pid}.jpg")
        img_out.save(out_img_path, quality=95)

        # 生成 YOLO TXT（无框则为空文件，作为负样本/背景）
        out_txt_path = os.path.join(OUTPUT_ROOT, "labels", split, f"{pid}.txt")
        with open(out_txt_path, "w", encoding="utf-8") as ftxt:
            if pid in patient_to_boxes:
                for (x_min, y_min, bw, bh) in patient_to_boxes[pid]:
                    line = build_yolo_line(
                        x_min, y_min, bw, bh,
                        orig_w, orig_h,
                        scale, px, py,
                    )
                    ftxt.write(line + "\n")
            # 如果 pid 没有框（target==0），则不写入任何行（保持空文件）

        stats[split] += 1

    print("\n转换完成：")
    print(f"  train images: {stats['train']}")
    print(f"  val images:   {stats['val']}")

    # 生成 data.yaml
    yaml_path = os.path.join(OUTPUT_ROOT, "data.yaml")
    yaml_content = f"""path: {os.path.abspath(OUTPUT_ROOT)}
train: images/train
val: images/val

nc: 1
names:
  0: pneumonia
"""
    with open(yaml_path, "w", encoding="utf-8") as fy:
        fy.write(yaml_content.strip())
    print(f"data.yaml 已生成：{yaml_path}")


if __name__ == "__main__":
    main()
