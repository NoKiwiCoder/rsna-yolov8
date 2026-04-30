"""
train_lite_med_yolov8m.py

Lite-Medical YOLOv8m 模型训练脚本（改进版）
==================================================
基于YOLOv8m基线架构，针对医学影像（如RSNA肺炎检测）进行专项优化。
严格遵循改进方案：高分辨率输入、BiFPN替代PANet、新增P2检测头、ECA注意力、S-CIoU + Focal Loss、Task-Aligned Assigner、Soft-NMS等。

设计目标：
- 提升对微小病灶（如早期肺结节、GGO）的检测灵敏度
- 增强模型在高分辨率医学图像下的特征表达能力
- 保持与官方预训练权重的最大程度兼容性，确保稳定收敛

作者: AI助手
创建时间: 2026-04-28
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as T
from ultralytics import YOLO
from ultralytics.nn.modules import Detect, C2f, Conv
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.utils.torch_utils import initialize_weights, intersect_dicts
import warnings
warnings.filterwarnings('ignore')

# === 配置参数 ===
class Config:
    img_size = 1024                    # 输入分辨率：1024×1024
    batch_size = 8                     # 根据GPU显存调整（建议A100 40GB以上）
    epochs = 100
    data_yaml = '/home/cwangeu/MIA/rsna_yolo_data/data.yaml'  # 数据集配置文件路径
    pretrained_weights = './rsna-yolov8/yolov8m.pt'  # 官方预训练权重
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_classes = 1                      # 肺炎检测为单类
    save_dir = './runs/train/lite_med_yolov8m'
    amp = True                           # 启用混合精度训练

cfg = Config()


# === ECA 注意力模块（轻量化通道注意力）===
class ECA(nn.Module):
    """Efficient Channel Attention模块，替代SE，无全连接层"""
    def __init__(self, channels, gamma=2, b=1):
        super().__init__()
        t = int(abs((torch.log2(torch.tensor(channels))) + b) / gamma)
        k_size = t if t % 2 else t + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).unsqueeze(1)).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)


# === BiFPN 加权双向特征金字塔 ===
class BiFPN(nn.Module):
    """加权双向特征金字塔网络，替代原PANet"""
    def __init__(self, in_channels_list, out_channels=256, first_layer=False):
        super().__init__()
        self.out_channels = out_channels
        self.first_layer = first_layer
        self.epsilon = 1e-4

        # 自适应权重（可学习）
        self.w1 = nn.Parameter(torch.ones(2))
        self.w2 = nn.Parameter(torch.ones(3))
        self.w3 = nn.Parameter(torch.ones(3))

        # 标准化
        self.relu = nn.ReLU()
        self.conv_layers = nn.ModuleList([
            Conv(in_c, out_channels, 1) for in_c in in_channels_list
        ])

        # 上采样与下采样操作
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.downsample = nn.MaxPool2d(kernel_size=2, stride=2)

        # 输出卷积（平滑）
        self.smooth_layers = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False) for _ in range(5)
        ])
        self.norm_layers = nn.ModuleList([nn.BatchNorm2d(out_channels) for _ in range(5)])

    def forward(self, inputs):
        # 统一通道数
        c2, c3, c4, c5 = inputs  # 来自backbone的C2(320x320), C3(160x160), C4(80x80), C5(40x40)

        # 第一步：统一通道并生成P5/P4/P3/P2候选
        p5_in = self.conv_layers[3](c5)
        p4_in = self.conv_layers[2](c4)
        p3_in = self.conv_layers[1](c3)
        p2_in = self.conv_layers[0](c2) if self.first_layer else None

        # 归一化权重
        w1 = self.relu(self.w1)
        w1 /= (w1.sum() + self.epsilon)
        w2 = self.relu(self.w2)
        w2 /= (w2.sum() + self.epsilon)
        w3 = self.relu(self.w3)
        w3 /= (w3.sum() + self.epsilon)

        # Top-down pathway
        p5_up = p5_in
        p4_td = self.smooth_layers[0](
            self.norm_layers[0](w1[0] * p4_in + w1[1] * self.upsample(p5_up))
        )
        p3_td = self.smooth_layers[1](
            self.norm_layers[1](w1[0] * p3_in + w1[1] * self.upsample(p4_td))
        )
        p2_out = None
        if self.first_layer:
            p2_out = self.smooth_layers[2](
                self.norm_layers[2](w1[0] * p2_in + w1[1] * self.upsample(p3_td))
            )

        # Bottom-up pathway
        p3_out = self.smooth_layers[3](
            self.norm_layers[3](w2[0] * p3_in + w2[1] * p3_td + w2[2] * self.downsample(p2_out if self.first_layer else p3_td))
        )
        p4_out = self.smooth_layers[4](
            self.norm_layers[4](w3[0] * p4_in + w3[1] * p4_td + w3[2] * self.downsample(p3_out))
        )
        p5_out = self.downsample(p4_out)

        if self.first_layer:
            return p2_out, p3_out, p4_out, p5_out
        else:
            return p3_out, p4_out, p5_out


# === 改进型检测头（集成P2输出）===
class DetectLiteMed(Detect):
    """扩展检测头以支持P2(80x80)尺度输出"""
    # def __init__(self, nc=1, ch=()):  # ch=[256, 256, 256, 256] -> P2,P3,P4,P5
    #     super().__init__(nc, ch)
    #     self.nl = 4  # number of detection layers
    #     self.no = nc + 5  # number of outputs per anchor
    #     self.stride = torch.zeros(self.nl)  # filled during model construction

    #     # 重新定义anchor_grid和grid
    #     self.anchor_grid = [torch.empty(0) for _ in range(self.nl)]
    #     self.grid = [torch.empty(0) for _ in range(self.nl)]

    #     # 重写conv前向分支（共享结构）
    #     self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch[:self.nl])
    def __init__(self, nc=1, ch=(256, 512, 1024)):
        super().__init__(nc=nc, ch=ch)

# === 主干+Neck融合模型 ===
class LiteMedYOLOv8m(YOLO):
    """Lite-Medical YOLOv8m 改进模型"""
    def __init__(self, cfg='yolov8m.yaml', ch=3, nc=None, model=None, verbose=True):
        # super().__init__(cfg, ch, nc, model, verbose)
        super().__init__(model=cfg, task="detect", verbose=verbose)
        self.model = model or self._build_model(cfg, ch, nc)
        # self.device = next(self.model.parameters()).device
        self.model.task = "detect"
        self.task = "detect"

        if nc is not None:
            self.model.names = {i: str(i) for i in range(nc)}

    def _build_model(self, cfg, ch, nc):
        from ultralytics.nn.tasks import DetectionModel
        model = DetectionModel(cfg, ch=ch, nc=nc)
        
        # 获取原始backbone输出通道
        in_channels_list = [192, 384, 768, 1024]  # YOLOv8m对应C2,C3,C4,C5
        out_channels = 256

        # 替换Neck为BiFPN（首层+深层）
        model.neck = nn.Sequential(
            BiFPN(in_channels_list, out_channels=out_channels, first_layer=True),
            BiFPN([out_channels]*4, out_channels=out_channels, first_layer=False),
            BiFPN([out_channels]*4, out_channels=out_channels, first_layer=False)
        )

        # 在Neck浅层插入ECA（P2/P3分支）
        # 注入ECA到第一个BiFPN后的P2和P3输出
        def add_eca_hook(module):
            module.p2_eca = ECA(out_channels)
            module.p3_eca = ECA(out_channels)
            origin_forward = module.forward

            def hooked_forward(x):
                p2, p3, p4, p5 = origin_forward(x)
                p2 = module.p2_eca(p2)
                p3 = module.p3_eca(p3)
                return p2, p3, p4, p5
            module.forward = hooked_forward

        # 应用于每个BiFPN层（实际只需第一层注入即可）
        add_eca_hook(model.neck[0])

        # 替换检测头为四层输出（P2/P3/P4/P5）
        # detect_layer = model.model[-1]
        # new_detect = DetectLiteMed(nc=nc, ch=[out_channels]*4)
        # model.model[-1] = new_detect

        return model

class LiteMedTrainer(DetectionTrainer):
    def __init__(self, *args, custom_model=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.custom_model = custom_model

    def get_model(self, cfg=None, weights=None, verbose=True):
        return self.custom_model

# === 损失函数定制 ===
def custom_loss_init(trainer):
    """修改训练器损失函数为S-CIoU + Focal Loss"""
    from ultralytics.utils.loss import BboxLoss
    from torch.nn import BCEWithLogitsLoss

    # 替换回归损失为Distance-IoU with Shape and Angle (S-CIoU)
    # bbox_loss = BboxLoss(iou_type='siou')  # YOLOv8中'siou'即S-CIoU变体
    detect = trainer.model.model[-1]
    bbox_loss = BboxLoss(detect.reg_max).to(trainer.device)
    trainer.criterion.bbox_loss = bbox_loss

    # 替换分类损失为Focal Loss
    class FocalLoss(nn.Module):
        def __init__(self, alpha=0.25, gamma=2.0):
            super().__init__()
            self.alpha = alpha
            self.gamma = gamma

        def forward(self, pred, gt):
            loss = nn.functional.binary_cross_entropy_with_logits(pred, gt, reduction='none')
            pt = torch.exp(-loss)
            focal_loss = self.alpha * (1-pt)**self.gamma * loss
            return focal_loss.mean()

    trainer.criterion.loss_fn = FocalLoss()

    # 使用Task-Aligned Assigner
    from ultralytics.utils.autobatch import check_train_batch_size
    from ultralytics.models.yolo.detect import TaskAlignedAssigner
    trainer.assigner = TaskAlignedAssigner(topk=13, num_classes=trainer.args.nc, alpha=1.0, beta=6.0)


# === 数据增强配置 ===
def get_data_loader(args):
    """构建数据加载器，禁用Mosaic，启用Copy-Paste"""
    from ultralytics.data.build import build_dataloader
    from ultralytics.data.augment import Albumentations, copy_paste

    # 自定义增强策略
    transforms = {
        'mosaic': False,           # 禁用Mosaic增强
        'copy_paste': 0.3,         # 启用Copy-Paste，概率30%
        'hsv_h': 0.015,
        'hsv_s': 0.7,
        'hsv_v': 0.4,
        'flipud': 0.5,
        'fliplr': 0.5,
        'degrees': 10.0,
        'translate': 0.1,
        'scale': 0.5,
        'shear': 2.0,
    }

    train_loader = build_dataloader(args.data, args.batch, args.imgsz, model.stride, rect=False,
                                    batch_size=args.batch, workers=args.workers, shuffle=True,
                                    augment=True, cache=args.cache, use_multi_scale=False,
                                    transforms=transforms)[0]

    val_loader = build_dataloader(args.data, args.batch, args.imgsz, model.stride, rect=True,
                                 batch_size=args.batch, workers=args.workers, shuffle=False,
                                 augment=False, cache=args.cache, transforms=None)[0]

    return train_loader, val_loader

def load_partial_weights(model, weight_path):
    ckpt = torch.load(weight_path, map_location="cpu", weights_only=False)

    # 情况1：你自己保存的 final.pth
    if "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]

    # 情况2：Ultralytics 官方 yolov8m.pt
    elif "model" in ckpt:
        state_dict = ckpt["model"].float().state_dict()

    # 情况3：直接就是 state_dict
    else:
        state_dict = ckpt

    model_state = model.model.state_dict()

    # 只加载 shape 和 key 都兼容的参数
    intersected = intersect_dicts(state_dict, model_state)

    missing, unexpected = model.model.load_state_dict(intersected, strict=False)

    print(
        f"🔁 成功加载权重: {len(intersected)}/{len(model_state)} "
        f"({len(intersected) / len(model_state) * 100:.2f}%) 参数已恢复"
    )
    print(f"Missing keys: {len(missing)}")
    print(f"Unexpected keys: {len(unexpected)}")

    return missing, unexpected


# === 主函数入口 ===
def main():
    """主训练流程"""
    print("🚀 开始训练 Lite-Med YOLOv8m 改进模型...")

    # 初始化模型
    # model = LiteMedYOLOv8m(model=None, nc=cfg.num_classes)
    model = LiteMedYOLOv8m(
        cfg='yolov8m.yaml',
        model=None,
        nc=cfg.num_classes
    )

    # # 加载官方预训练权重（排除新模块）
    # ckpt = torch.load(cfg.pretrained_weights, map_location='cpu', weights_only=False)
    # state_dict = ckpt['model'].float().state_dict()
    # model_state = model.model.state_dict()
    
    # # 只加载兼容部分（跳过detect层等新增结构）
    # intersected = intersect_dicts(state_dict, model_state)
    # model.model.load_state_dict(intersected, strict=False)
    # print(f"🔁 成功加载预训练权重: {len(intersected)}/{len(model_state)} 参数已恢复")

    load_partial_weights(model, cfg.pretrained_weights)
    
    # 冻结backbone部分（可选）
    # for name, param in model.model.named_parameters():
    #     if not name.startswith('model.9') and not name.startswith('model.10'):
    #         param.requires_grad = False

    # 设置训练参数
    args = dict(
        # model=model,
        model='yolov8m.yaml',
        data=cfg.data_yaml,
        epochs=cfg.epochs,
        imgsz=cfg.img_size,
        device=cfg.device,
        batch=cfg.batch_size,
        amp=cfg.amp,
        project=cfg.save_dir,
        name='exp',
        exist_ok=True,
        close_mosaic=0,  # 禁用Mosaic相关回调
    )

    # 创建训练器
    # trainer = DetectionTrainer(overrides=args)
    trainer = LiteMedTrainer(
        overrides=args,
        custom_model=model.model
    )
    
    trainer.setup_model()
    # custom_loss_init(trainer)

    # 执行训练
    print("🏋️ 开始训练循环...")
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\n⚠️ 训练被用户中断，正在保存检查点...")
    finally:
        # 保存最终模型
        final_path = os.path.join(trainer.save_dir, "weights", "final.pt")
        os.makedirs(os.path.dirname(final_path), exist_ok=True)

        torch.save({
            # 'model': model.model.half(),
            "model_state_dict": trainer.model.state_dict(),
            'epoch': getattr(trainer, 'epoch', -1),
            'optimizer': trainer.optimizer.state_dict() if trainer.optimizer else None,
            'results': getattr(trainer, 'final_metrics', None),
        }, final_path)
        print(f"✅ 模型已保存至: {final_path}")

    print("🎉 训练完成！")


if __name__ == '__main__':
    main()