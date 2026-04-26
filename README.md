
# RSNA肺结节检测 - YOLOv8复现指南

## 项目概述
本项目基于 Ultralytics YOLOv8 框架，使用 RSNA 数据集进行肺结节检测模型的训练与评估。主要目标是实现高精度的肺结节识别，并探索 Inner-CIoU 损失函数的效果。

---

## 环境设置
1. **创建并激活 conda 环境**
    ```bash
    conda create -n yolomed_env python=3.10
    conda activate yolomed_env
    ```

2. **一键安装所有依赖（可选）**
    ```bash
    pip install -r requirements.txt
    ```

3. **安装 PyTorch（CUDA 13.0）看自己cuda版本安装，50系装preview版本**
    ```bash
    pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu130
    ```

4. **安装 Ultralytics**
    ```bash
    pip install ultralytics
    ```

5. **验证安装**
    ```bash
    python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
    ```
> 版本什么的对不上就看报错自己装一装包把。
---

## 数据下载与准备

1. **从 Kaggle 下载数据集**
    - 先在根目录下创建一个dataset文件夹
    - 直接下压缩包到dataset目录下解压缩就行了
    - 链接：https://www.kaggle.com/competitions/rsna-pneumonia-detection-challenge/data
2. **数据转换**
    - 将原始 DICOM 图像转换为 YOLOv8 格式，执行以下脚本：
      ```bash
      python rsna_to_yolo.py
      ```
3. **数据集结构**
    ```text
    dataset/
    ├── rsna-pneumonia-detection-challenge/
    │   ├── stage_2_train_images/
    │   ├── stage_2_train_labels.csv
    │   └── ...
    └── rsna_yolo/
         ├── images/
         │   └── train/
         └── labels/
              └── train/
    ```
4. **创建 data.yaml 配置文件，修改成自己的路径**
    例如
    ```yaml
    path: C:\Users\Admin\Desktop\MedicalProject\dataset\rsna_yolo
    train: images/train
    val: images/val

    nc: 1
    names:
    0: pneumonia
    ```

---

## 模型训练
1. **训练脚本**
    > ⚙️ **所有训练参数请在 `train_yolov8_ultralytics.py` 脚本中修改！**
    
    ```python
    # 训练参数
    model.train(
        data='dataset/rsna_yolo/data.yaml',
        epochs=80,
        imgsz=512,
        batch=8,
        device='0',
        workers=0,  # 并发数，Windows环境下建议设为0，配置好可以增加
        patience=20,
        project='runs_rsna',
        name='ema_innerciou',
        exist_ok=True,
        verbose=True
    )
    ```
    
    **运行训练脚本：**
    ```bash
    python train_yolov8_ultralytics.py
    ```
2. **训练参数说明**
    - `epochs=80`：训练轮数（默认上限，实际可能提前停止）
    - `imgsz=512`：输入图像尺寸
    - `batch=8`：批次大小
    - `workers=0`：数据加载进程数（Windows环境下避免崩溃）
    - `patience=20`：早停机制，20个 epoch 无提升则停止
> **实验规范：** 每次不同实验必须修改 `train_yolov8_ultralytics.py` 中 `model.train(..., name=...)` 的 `name` 参数，确保结果目录唯一，便于追踪和对比。
---

## 模型评估
### 当前模型架构与优化方法

本轮实验采用如下模型架构与优化策略（所有实现细节见 `train_yolov8_ultralytics.py`）：

- **注意力机制增强**：自定义并注册了 EMA（通道注意力）模块，在 Head 多个特征融合节点后插入，动态调整特征通道权重，提升小目标特征提取能力。
- **损失函数改进**：重写 bbox_iou，采用 Inner-CIoU 替换原 CIoU，优化边界框中心点距离惩罚，提升小目标定位精度。
- **模型结构调整**：在 YOLOv8 Head 的 C2f 层后多处插入 EMA 层，增强多尺度特征融合的有效性。
- **训练参数与数据增强**：其余参数与数据增强采用 YOLOv8 默认配置。

> **注意：** 评估结果与本节所述架构和优化方法一一对应。若后续更改模型结构、损失函数或训练参数，请同步更新本节内容，确保实验结果可追溯。

1. **评估指标**
    - `mAP50`：平均精度均值（IoU≥0.5），核心评估指标
    - `mAP50-95`：严格评估标准（IoU从0.5到0.95）
    - `Recall`：召回率，避免漏诊的关键指标
2. **评估结果**
    ```text
    Epoch 80/80
    mAP50: 0.303
    mAP50-95: 0.123
    Recall: 0.323
    ```
3. **结果分析**
    - mAP50 达到 0.303，属于中等偏上水平
    - mAP50-95 偏低是正常现象（肺结节小目标特性）
    - 模型已收敛，未出现过拟合

---

## 可视化结果
训练完成后，在 `runs_rsna/ema_innerciou/` 目录下生成：
- `results.png`：训练指标变化图
- `confusion_matrix.png`：混淆矩阵
- `val_batch*_pred.jpg`：验证集预测结果可视化

---

## 注意事项
- Windows 环境下建议将 workers 设为 0，避免 DataLoader 崩溃
- 如遇内存问题，可适当降低 batch 大小
- 训练过程中监控 GPU 内存使用情况

---

## 参考文献
- [YOLOv8官方文档](https://docs.ultralytics.com/)
- [RSNA肺结节检测数据集](https://www.rsna.org/education/ai-resources-and-training/ai-image-challenge/rsna-pneumonia-detection-challenge-2018)