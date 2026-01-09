"""
简单的分割效果可视化测试脚本
用于快速检查模型的分割质量
"""

import argparse
import os
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import cv2
from pathlib import Path

# 添加项目路径
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.FlowFormer import build_flowformer
from core.segmentation import MotionSegmentationModel
from core.utils.flow_viz import flow_to_image


def load_image(path, max_size=960):
    """加载图像并转换为 tensor，限制最大尺寸"""
    img = Image.open(path).convert('RGB')
    
    # 限制最大尺寸
    W, H = img.size
    if max(H, W) > max_size:
        scale = max_size / max(H, W)
        new_W = int(W * scale)
        new_H = int(H * scale)
        img = img.resize((new_W, new_H), Image.BILINEAR)
        print(f"  图像从 {W}x{H} 缩放到 {new_W}x{new_H}")
    
    img = np.array(img).astype(np.float32)
    img = torch.from_numpy(img).permute(2, 0, 1)  # [3, H, W]
    return img


def pad_to_multiple(img, multiple=8):
    """填充图像到指定倍数"""
    _, H, W = img.shape
    pad_h = (multiple - H % multiple) % multiple
    pad_w = (multiple - W % multiple) % multiple
    if pad_h > 0 or pad_w > 0:
        img = F.pad(img.unsqueeze(0), (0, pad_w, 0, pad_h), mode='replicate').squeeze(0)
    return img, (H, W)


def get_slot_colors(num_slots):
    """生成 slot 颜色"""
    colors = [
        [255, 0, 0],      # 红
        [0, 255, 0],      # 绿
        [0, 0, 255],      # 蓝
        [255, 255, 0],    # 黄
        [255, 0, 255],    # 品红
        [0, 255, 255],    # 青
        [255, 128, 0],    # 橙
        [128, 0, 255],    # 紫
        [0, 128, 128],    # 蓝绿
        [128, 128, 0],    # 橄榄
    ]
    return np.array(colors[:num_slots], dtype=np.float32)


def visualize_masks(masks, colors):
    """
    可视化分割 masks
    masks: [K, H, W]
    """
    K, H, W = masks.shape
    seg_map = np.zeros((H, W, 3), dtype=np.float32)
    
    for k in range(K):
        mask_k = masks[k]
        seg_map += mask_k[:, :, None] * colors[k]
    
    seg_map = np.clip(seg_map, 0, 255).astype(np.uint8)
    return seg_map


def visualize_masks_separate(masks, colors):
    """
    分别可视化每个 slot 的 mask
    返回一个拼接的图像
    """
    K, H, W = masks.shape
    
    # 创建每个 slot 的可视化
    slot_vis = []
    for k in range(K):
        mask_k = masks[k]
        # 灰度图
        gray = (mask_k * 255).astype(np.uint8)
        # 转为彩色
        colored = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
        slot_vis.append(colored)
    
    # 拼接成一行
    combined = np.concatenate(slot_vis, axis=1)
    return combined


def main():
    parser = argparse.ArgumentParser(description='测试分割效果')
    parser.add_argument('--model', type=str, required=True, help='模型权重路径')
    parser.add_argument('--image1', type=str, required=True, help='第一帧图像')
    parser.add_argument('--image2', type=str, required=True, help='第二帧图像')
    parser.add_argument('--output', type=str, default='test_output', help='输出目录')
    parser.add_argument('--num_slots', type=int, default=7, help='slot 数量')
    parser.add_argument('--motion_model', type=str, default='hybrid', 
                        choices=['dense', 'cnn', 'hybrid'], help='运动模型类型')
    parser.add_argument('--max_size', type=int, default=640, help='图像最大尺寸，防止显存溢出')
    parser.add_argument('--device', type=str, default='cuda')
    
    args = parser.parse_args()
    
    # 创建输出目录
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"加载模型: {args.model}")
    
    # 加载配置
    from configs.sintel import get_cfg
    cfg = get_cfg()
    
    # 构建模型
    flowformer = build_flowformer(cfg)
    model = MotionSegmentationModel(
        flowformer_model=flowformer,
        num_slots=args.num_slots,
        slot_dim=64,
        hidden_dim=128,
        num_iterations=3,
        motion_model=args.motion_model,
        feature_scale='1/8'
    )
    
    # 加载权重
    state_dict = torch.load(args.model, map_location=args.device)
    # 处理可能的 module. 前缀
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k[7:]: v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    
    model.to(args.device)
    model.eval()
    
    print(f"加载图像: {args.image1}, {args.image2}")
    
    # 加载图像（限制尺寸）
    image1 = load_image(args.image1, max_size=args.max_size)
    image2 = load_image(args.image2, max_size=args.max_size)
    
    # 保存原始尺寸
    _, H_orig, W_orig = image1.shape
    
    # 填充到 8 的倍数
    image1_padded, _ = pad_to_multiple(image1)
    image2_padded, _ = pad_to_multiple(image2)
    
    # 添加 batch 维度
    image1_batch = image1_padded.unsqueeze(0).to(args.device)
    image2_batch = image2_padded.unsqueeze(0).to(args.device)
    
    print("运行推理...")
    
    with torch.no_grad():
        output = model(image1_batch, image2_batch, return_flow=True)
    
    # 提取结果
    masks = output['masks'][0, :, :H_orig, :W_orig].cpu().numpy()  # [K, H, W]
    flow_recon = output['flow_recon'][0, :, :H_orig, :W_orig].cpu().numpy()  # [2, H, W]
    
    if 'flow_pred' in output:
        flow_pred = output['flow_pred'][:, :, :H_orig, :W_orig].cpu().numpy()
    elif 'flow_predictions' in output:
        flow_pred = output['flow_predictions'][-1][0, :, :H_orig, :W_orig].cpu().numpy()
    else:
        flow_pred = None
    
    motion_info = output.get('motion_info', {})
    
    print(f"\n=== 结果统计 ===")
    print(f"Masks shape: {masks.shape}")
    print(f"Flow recon range: [{flow_recon.min():.2f}, {flow_recon.max():.2f}]")
    if flow_pred is not None:
        print(f"Flow pred range: [{flow_pred.min():.2f}, {flow_pred.max():.2f}]")
    
    # 全局运动信息
    if 'global_flow' in motion_info:
        global_flow = motion_info['global_flow'][0].cpu().numpy()
        print(f"Global flow range: [{global_flow.min():.2f}, {global_flow.max():.2f}]")
    
    if 'homography_params' in motion_info:
        h_params = motion_info['homography_params'][0].cpu().numpy()
        print(f"Homography params: {h_params}")
    
    # 每个 slot 的统计
    print(f"\n=== Slot 统计 ===")
    K = masks.shape[0]
    slot_flows_np = output['slot_flows'][0].cpu().numpy()  # [K, 2, H, W]
    for k in range(K):
        area = masks[k].sum() / (H_orig * W_orig) * 100
        flow_k = slot_flows_np[k]
        flow_mag = np.sqrt(flow_k[0]**2 + flow_k[1]**2).mean()
        print(f"Slot {k}: 面积={area:.1f}%, 平均光流幅度={flow_mag:.2f}")
    
    # 生成可视化
    print(f"\n保存结果到: {output_dir}")
    
    colors = get_slot_colors(K)
    
    # 1. 原始图像
    img1_np = image1.permute(1, 2, 0).numpy().astype(np.uint8)
    cv2.imwrite(str(output_dir / 'image1.png'), img1_np[:, :, ::-1])
    
    # 2. 分割结果 (彩色叠加)
    seg_map = visualize_masks(masks, colors)
    cv2.imwrite(str(output_dir / 'segmentation.png'), seg_map[:, :, ::-1])
    
    # 3. 分割叠加到原图
    overlay = img1_np.astype(np.float32) * 0.5 + seg_map.astype(np.float32) * 0.5
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    cv2.imwrite(str(output_dir / 'overlay.png'), overlay[:, :, ::-1])
    
    # 4. 每个 slot 的 mask (热力图)
    slot_heatmaps = visualize_masks_separate(masks, colors)
    cv2.imwrite(str(output_dir / 'slot_masks.png'), slot_heatmaps)
    
    # 5. 重建光流
    flow_recon_vis = flow_to_image(flow_recon.transpose(1, 2, 0))
    cv2.imwrite(str(output_dir / 'flow_recon.png'), flow_recon_vis[:, :, ::-1])
    
    # 6. FlowFormer 预测的光流
    if flow_pred is not None:
        # 确保维度正确
        if flow_pred.ndim == 3:
            flow_pred_np = flow_pred.transpose(1, 2, 0)  # [2, H, W] -> [H, W, 2]
        else:
            flow_pred_np = flow_pred[0].transpose(1, 2, 0)  # [1, 2, H, W] -> [H, W, 2]
        flow_pred_vis = flow_to_image(flow_pred_np)
        cv2.imwrite(str(output_dir / 'flow_pred.png'), flow_pred_vis[:, :, ::-1])
    
    # 7. 找出前景 (使用模型的方法)
    with torch.no_grad():
        slot_flows_tensor = output['slot_flows']
        masks_tensor = output['masks'][:, :, :H_orig, :W_orig]
        fg_mask_tensor, bg_idx = model.identify_foreground(
            masks_tensor, slot_flows_tensor, motion_info
        )
    
    fg_mask = fg_mask_tensor[0, 0].cpu().numpy()
    fg_mask_img = (fg_mask * 255).astype(np.uint8)
    cv2.imwrite(str(output_dir / 'foreground.png'), fg_mask_img)
    
    print(f"\n背景 Slot: {bg_idx[0].item()}")
    
    # 8. 前景叠加
    fg_overlay = img1_np.copy()
    fg_overlay[fg_mask > 0.5] = fg_overlay[fg_mask > 0.5] * 0.5 + np.array([0, 255, 0]) * 0.5
    cv2.imwrite(str(output_dir / 'foreground_overlay.png'), fg_overlay[:, :, ::-1])
    
    print("\n=== 输出文件 ===")
    print(f"  image1.png          - 原始图像")
    print(f"  segmentation.png    - 分割结果 (彩色)")
    print(f"  overlay.png         - 分割叠加到原图")
    print(f"  slot_masks.png      - 每个 slot 的热力图")
    print(f"  flow_recon.png      - 重建的光流")
    print(f"  flow_pred.png       - FlowFormer 预测的光流")
    print(f"  foreground.png      - 前景 mask")
    print(f"  foreground_overlay.png - 前景叠加到原图")
    
    print("\n完成!")


if __name__ == '__main__':
    main()
