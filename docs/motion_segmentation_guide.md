# FlowFormer 运动分割指南

## 概述

本模块在 FlowFormer 的基础上实现了自监督运动分割，通过 Slot Attention 机制将图像分割为具有一致运动模式的区域。

## 核心思想

**物体 = 具有一致运动模式的区域**

- 利用 FlowFormer 强大的 Encoder 特征（包含上下文和纹理信息）
- 通过 Slot Attention 进行无监督的物体发现
- 使用光流重建损失来监督分割头，无需分割标签

## 模块结构

```
core/segmentation/
├── __init__.py
├── wrapper.py           # FlowFormer Wrapper，复用 Encoder 特征
├── slot_attention.py    # Slot Attention Head
├── motion_decoder.py    # Motion Decoder，从 Slot 解码运动参数
├── segmentation_model.py # 完整模型
└── losses.py            # 损失函数
```

## 训练流程

### Phase 1: 热身阶段 (Freeze Backbone, Train Head)

```bash
python train_motion_segmentation.py \
    --name exp1 \
    --stage things \
    --flowformer_ckpt checkpoints/flowformer_things.pth \
    --phase1_steps 50000 \
    --phase1_lr 2e-4 \
    --num_slots 7 \
    --batch_size 4
```

**策略:**
- 冻结 FlowFormer 的 Encoder 和 Flow Head
- 只训练 SlotAttention 和 MotionDecoder
- Loss = MSE(Flow_Recon, Flow_Pred.detach())
- 把 FlowFormer 当作教师，Segment Head 尝试用 K 个分层的运动去解释老师输出的光流场

### Phase 2: 联合优化 (Fine-tune)

```bash
python train_motion_segmentation.py \
    --name exp1 \
    --stage things \
    --flowformer_ckpt checkpoints/flowformer_things.pth \
    --skip_phase1 \
    --phase1_ckpt checkpoints/motion_seg/exp1/phase1_final.pth \
    --phase2_steps 100000 \
    --phase2_lr 1e-4 \
    --unfreeze_layers 2 \
    --flow_weight 1.0 \
    --seg_weight 0.5
```

**策略:**
- 解冻 Encoder 的后几层
- 联合优化: Loss = Weight_Flow * L_Flow + Weight_Seg * L_Recon
- 让 Encoder 学出的特征更适合聚类

## 推理

### 基本推理

```python
from inference_motion_segmentation import MotionSegmentationInference, load_image

# 加载模型
inference = MotionSegmentationInference(
    model_path='checkpoints/motion_seg/exp1/final_model.pth',
    flowformer_cfg='things',
    device='cuda'
)

# 加载图像
image1 = load_image('frame1.png')
image2 = load_image('frame2.png')

# 预测
result = inference.predict(image1, image2)
# result 包含:
#   - flow: 光流
#   - masks: K 个分割 masks
#   - fg_mask: 前景 mask
#   - bg_idx: 背景 slot 索引
```

### 变化检测

```python
# 变化检测
change_mask, change_objects = inference.detect_changes(image1, image2, threshold=0.5)

# change_objects 是一个列表，每个元素包含:
#   - slot_id: slot 索引
#   - mask: 该物体的 mask
#   - bbox: 边界框 (x_min, y_min, x_max, y_max)
#   - area: 面积
#   - motion_params: 运动参数
```

### 命令行推理

```bash
python inference_motion_segmentation.py \
    --model checkpoints/motion_seg/exp1/final_model.pth \
    --image1 frame1.png \
    --image2 frame2.png \
    --output_dir output \
    --cfg things
```

## 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| num_slots | 7 | Slot 数量，代表最大物体数 |
| slot_dim | 64 | Slot 特征维度 |
| slot_iterations | 3 | Slot Attention 迭代次数 |
| motion_model | affine | 运动模型类型 (affine/quadratic) |
| feature_scale | 1/8 | 特征图分辨率 |

## 损失函数

1. **光流重建损失 (L_recon)**: 核心损失，用重建的光流与目标光流的差异来监督
2. **Mask 正则化损失**: 包括熵损失、面积平衡损失、平滑损失
3. **运动一致性损失**: 鼓励同一 slot 内的像素具有相似运动

## 背景识别

背景 slot 的识别基于两个特征:
- **面积最大**: 背景通常占据图像的大部分
- **运动最小**: 背景通常是静止的或运动很小

```python
bg_score = areas_norm - motion_norm  # 面积大、运动小的得分高
```

## 注意事项

1. **显存占用**: 建议使用 1/8 分辨率特征图，显存占用可控
2. **Slot 数量**: 建议 5-10 个，太少可能无法分割所有物体，太多可能导致过分割
3. **训练数据**: 推荐使用包含多个运动物体的数据集 (如 FlyingThings3D)
4. **预训练权重**: 必须使用预训练的 FlowFormer 权重，否则特征质量不足
