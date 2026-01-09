"""
Motion Segmentation 推理脚本
用于变化检测和物体分割
"""

import argparse
import os
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import cv2
from pathlib import Path

from core.FlowFormer import build_flowformer
from core.segmentation import MotionSegmentationModel
from core.utils.flow_viz import flow_to_image


def load_image(path, size=None):
    """加载图像"""
    img = Image.open(path).convert('RGB')
    if size is not None:
        img = img.resize(size, Image.BILINEAR)
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


def visualize_masks(masks, num_slots):
    """可视化分割 masks"""
    # masks: [K, H, W]
    K, H, W = masks.shape
    
    # 生成颜色映射
    colors = np.array([
        [255, 0, 0],    # 红
        [0, 255, 0],    # 绿
        [0, 0, 255],    # 蓝
        [255, 255, 0],  # 黄
        [255, 0, 255],  # 品红
        [0, 255, 255],  # 青
        [128, 128, 0],  # 橄榄
        [128, 0, 128],  # 紫
        [0, 128, 128],  # 蓝绿
        [255, 128, 0],  # 橙
    ], dtype=np.float32)
    
    # 创建彩色分割图
    seg_map = np.zeros((H, W, 3), dtype=np.float32)
    for k in range(min(K, len(colors))):
        mask_k = masks[k].cpu().numpy()
        color = colors[k % len(colors)]
        seg_map += mask_k[:, :, None] * color
    
    seg_map = np.clip(seg_map, 0, 255).astype(np.uint8)
    return seg_map


def visualize_slot_flows(slot_flows, masks):
    """可视化每个 slot 的运动场"""
    # slot_flows: [K, 2, H, W]
    # masks: [K, H, W]
    K = slot_flows.shape[0]
    
    flow_vis_list = []
    for k in range(K):
        flow_k = slot_flows[k].cpu().numpy().transpose(1, 2, 0)  # [H, W, 2]
        flow_img = flow_to_image(flow_k)
        
        # 用 mask 加权
        mask_k = masks[k].cpu().numpy()[:, :, None]
        flow_img = (flow_img * mask_k).astype(np.uint8)
        flow_vis_list.append(flow_img)
    
    return flow_vis_list


class MotionSegmentationInference:
    """运动分割推理类"""
    
    def __init__(self, model_path, flowformer_cfg, device='cuda'):
        self.device = device
        
        # 加载配置
        if flowformer_cfg == 'things':
            from configs.things import get_cfg
        elif flowformer_cfg == 'sintel':
            from configs.sintel import get_cfg
        else:
            from configs.default import get_cfg
        
        self.cfg = get_cfg()
        
        # 构建模型
        flowformer = build_flowformer(self.cfg)
        self.model = MotionSegmentationModel(
            flowformer_model=flowformer,
            num_slots=7,
            slot_dim=64,
            hidden_dim=128,
            num_iterations=3,
            motion_model='affine',
            feature_scale='1/8'
        )
        
        # 加载权重
        state_dict = torch.load(model_path, map_location=device)
        if list(state_dict.keys())[0].startswith('module.'):
            state_dict = {k[7:]: v for k, v in state_dict.items()}
        self.model.load_state_dict(state_dict)
        
        self.model.to(device)
        self.model.eval()
        
    @torch.no_grad()
    def predict(self, image1, image2):
        """
        预测光流和分割
        
        Args:
            image1: [3, H, W] 或 [B, 3, H, W] 第一帧
            image2: [3, H, W] 或 [B, 3, H, W] 第二帧
            
        Returns:
            dict containing:
                - flow: [B, 2, H, W] 光流
                - masks: [B, K, H, W] 分割 masks
                - fg_mask: [B, 1, H, W] 前景 mask
                - bg_idx: [B] 背景 slot 索引
        """
        # 添加 batch 维度
        if image1.dim() == 3:
            image1 = image1.unsqueeze(0)
            image2 = image2.unsqueeze(0)
            
        image1 = image1.to(self.device)
        image2 = image2.to(self.device)
        
        # 填充到 8 的倍数
        _, _, H, W = image1.shape
        image1_padded, orig_size = pad_to_multiple(image1[0])
        image2_padded, _ = pad_to_multiple(image2[0])
        image1_padded = image1_padded.unsqueeze(0)
        image2_padded = image2_padded.unsqueeze(0)
        
        # 推理
        output = self.model(image1_padded, image2_padded, return_flow=True)
        
        # 裁剪回原始尺寸
        H_orig, W_orig = orig_size
        flow = output['flow_pred'][:, :, :H_orig, :W_orig] if 'flow_pred' in output else output['flow_recon'][:, :, :H_orig, :W_orig]
        masks = output['masks'][:, :, :H_orig, :W_orig]
        
        # 识别背景
        bg_idx, fg_mask = self.model.identify_background(
            masks, output['motion_params']
        )
        fg_mask = fg_mask[:, :, :H_orig, :W_orig]
        
        return {
            'flow': flow,
            'flow_recon': output['flow_recon'][:, :, :H_orig, :W_orig],
            'masks': masks,
            'fg_mask': fg_mask,
            'bg_idx': bg_idx,
            'slots': output['slots'],
            'motion_params': output['motion_params']
        }
    
    def detect_changes(self, image1, image2, threshold=0.5):
        """
        变化检测
        
        Args:
            image1, image2: 输入图像
            threshold: 前景阈值
            
        Returns:
            change_mask: [H, W] 变化区域 mask
            change_objects: list of dict, 每个变化物体的信息
        """
        result = self.predict(image1, image2)
        
        # 前景 mask
        fg_mask = result['fg_mask'][0, 0].cpu().numpy()
        change_mask = (fg_mask > threshold).astype(np.uint8)
        
        # 分析每个前景 slot
        masks = result['masks'][0].cpu().numpy()  # [K, H, W]
        motion_params = result['motion_params'][0].cpu().numpy()  # [K, P]
        bg_idx = result['bg_idx'][0].item()
        
        change_objects = []
        K = masks.shape[0]
        
        for k in range(K):
            if k == bg_idx:
                continue
                
            mask_k = masks[k]
            area = mask_k.sum()
            
            if area > 100:  # 过滤太小的区域
                # 计算边界框
                coords = np.where(mask_k > 0.3)
                if len(coords[0]) > 0:
                    y_min, y_max = coords[0].min(), coords[0].max()
                    x_min, x_max = coords[1].min(), coords[1].max()
                    
                    change_objects.append({
                        'slot_id': k,
                        'mask': mask_k,
                        'bbox': (x_min, y_min, x_max, y_max),
                        'area': area,
                        'motion_params': motion_params[k]
                    })
        
        return change_mask, change_objects


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help="path to model checkpoint")
    parser.add_argument('--image1', type=str, required=True, help="path to first image")
    parser.add_argument('--image2', type=str, required=True, help="path to second image")
    parser.add_argument('--output_dir', type=str, default='output', help="output directory")
    parser.add_argument('--cfg', type=str, default='things', help="FlowFormer config")
    parser.add_argument('--device', type=str, default='cuda')
    
    args = parser.parse_args()
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载模型
    print(f"Loading model from {args.model}")
    inference = MotionSegmentationInference(args.model, args.cfg, args.device)
    
    # 加载图像
    print(f"Loading images: {args.image1}, {args.image2}")
    image1 = load_image(args.image1)
    image2 = load_image(args.image2)
    
    # 推理
    print("Running inference...")
    result = inference.predict(image1, image2)
    
    # 变化检测
    change_mask, change_objects = inference.detect_changes(image1, image2)
    
    # 保存结果
    print(f"Saving results to {output_dir}")
    
    # 1. 保存光流
    flow = result['flow'][0].cpu().numpy().transpose(1, 2, 0)
    flow_img = flow_to_image(flow)
    cv2.imwrite(str(output_dir / 'flow.png'), flow_img[:, :, ::-1])
    
    # 2. 保存分割 masks
    masks = result['masks'][0]
    seg_map = visualize_masks(masks, masks.shape[0])
    cv2.imwrite(str(output_dir / 'segmentation.png'), seg_map[:, :, ::-1])
    
    # 3. 保存前景 mask
    fg_mask = result['fg_mask'][0, 0].cpu().numpy()
    fg_mask_img = (fg_mask * 255).astype(np.uint8)
    cv2.imwrite(str(output_dir / 'foreground.png'), fg_mask_img)
    
    # 4. 保存变化检测结果
    change_mask_img = (change_mask * 255).astype(np.uint8)
    cv2.imwrite(str(output_dir / 'change_mask.png'), change_mask_img)
    
    # 5. 叠加可视化
    img1_np = image1.permute(1, 2, 0).numpy().astype(np.uint8)
    overlay = img1_np.copy()
    overlay[change_mask > 0] = overlay[change_mask > 0] * 0.5 + np.array([0, 255, 0]) * 0.5
    cv2.imwrite(str(output_dir / 'overlay.png'), overlay[:, :, ::-1])
    
    print(f"Found {len(change_objects)} change objects")
    for i, obj in enumerate(change_objects):
        print(f"  Object {i+1}: slot={obj['slot_id']}, area={obj['area']:.0f}, bbox={obj['bbox']}")
    
    print("Done!")


if __name__ == '__main__':
    main()
