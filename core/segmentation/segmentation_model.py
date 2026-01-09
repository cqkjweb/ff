"""
Motion Segmentation Model - 完整的多任务模型
结合 FlowFormer + Slot Attention + Motion Decoder

重新设计版本:
- 支持复杂的非对齐视角变化
- 使用混合运动模型 (全局单应性 + 局部残差)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .wrapper import FlowFormerWrapper
from .slot_attention import SlotAttentionHead
from .motion_decoder import MotionDecoder


class MotionSegmentationModel(nn.Module):
    """
    完整的运动分割模型
    
    Pipeline:
    1. FlowFormer Encoder 提取特征
    2. Slot Attention 聚类特征到 K 个 slots
    3. Motion Decoder 从 slots 解码运动
    4. 使用 masks 和运动重建光流
    """
    
    def __init__(
        self,
        flowformer_model,
        num_slots: int = 7,
        slot_dim: int = 64,
        hidden_dim: int = 128,
        num_iterations: int = 3,
        motion_model: str = 'hybrid',  # 'dense', 'cnn', 'hybrid'
        feature_scale: str = '1/8'
    ):
        """
        Args:
            flowformer_model: 预训练的 FlowFormer 模型
            num_slots: slot 数量 (K)
            slot_dim: slot 特征维度
            hidden_dim: MLP 隐藏层维度
            num_iterations: Slot Attention 迭代次数
            motion_model: 运动模型类型
            feature_scale: 特征图分辨率
        """
        super().__init__()
        
        self.num_slots = num_slots
        self.motion_model_type = motion_model
        
        # FlowFormer Wrapper
        self.flowformer_wrapper = FlowFormerWrapper(
            flowformer_model, 
            feature_scale=feature_scale
        )
        
        # Slot Attention Head
        input_dim = self.flowformer_wrapper.context_dim  # 256
        self.slot_attention_head = SlotAttentionHead(
            input_dim=input_dim,
            num_slots=num_slots,
            slot_dim=slot_dim,
            hidden_dim=hidden_dim,
            num_iterations=num_iterations
        )
        
        # Motion Decoder
        self.motion_decoder = MotionDecoder(
            slot_dim=slot_dim,
            hidden_dim=hidden_dim,
            context_dim=input_dim,
            motion_model=motion_model
        )

    def forward(self, image1, image2, return_flow=True):
        """
        前向传播
        
        Args:
            image1: [B, 3, H, W] 第一帧图像 (0-255)
            image2: [B, 3, H, W] 第二帧图像 (0-255)
            return_flow: 是否返回 FlowFormer 的光流预测
            
        Returns:
            dict containing:
                - flow_pred: FlowFormer 预测的光流
                - flow_recon: 重建的光流
                - masks: [B, K, H, W] 分割 masks
                - slots: [B, K, D] slot 特征
                - slot_flows: [B, K, 2, H, W] 每个 slot 的光流
                - motion_info: 运动参数信息
        """
        B, _, H, W = image1.shape
        
        # 1. 提取 FlowFormer 特征
        ff_output = self.flowformer_wrapper(image1, image2)
        context_features = ff_output['context_features']  # [B, 256, H/8, W/8]
        feature_size = ff_output['feature_size']
        
        # 2. Slot Attention
        slots, masks, attn_masks = self.slot_attention_head(
            context_features, feature_size
        )
        
        # 3. Motion Decoder - 重建光流
        H8, W8 = H // 8, W // 8
        
        flow_recon, slot_flows, motion_info = self.motion_decoder(
            slots=slots,
            masks=attn_masks,
            target_size=(H8, W8),
            full_size=(H, W),
            context_feat=context_features
        )
        
        # 上采样重建光流到原始分辨率
        flow_recon_full = F.interpolate(
            flow_recon, size=(H, W), mode='bilinear', align_corners=False
        )
        
        # 上采样 masks 到原始分辨率
        masks_full = F.interpolate(
            masks, size=(H, W), mode='bilinear', align_corners=False
        )
        masks_full = F.softmax(masks_full, dim=1)
        
        output = {
            'flow_recon': flow_recon_full,
            'flow_recon_8x': flow_recon,
            'masks': masks_full,
            'masks_8x': attn_masks,
            'slots': slots,
            'slot_flows': slot_flows,
            'motion_info': motion_info
        }
        
        if return_flow:
            flow_predictions = ff_output['flow_predictions']
            if self.training:
                output['flow_predictions'] = flow_predictions
            else:
                output['flow_pred'] = flow_predictions[-1] if isinstance(flow_predictions, list) else flow_predictions[0]
                
        return output
    
    def freeze_flowformer(self):
        """Phase 1: 冻结整个 FlowFormer"""
        self.flowformer_wrapper.freeze_encoder()
        self.flowformer_wrapper.freeze_flow_head()
        
    def unfreeze_encoder_partial(self, num_layers=2):
        """Phase 2: 解冻 Encoder 的后几层"""
        self.flowformer_wrapper.unfreeze_encoder_layers(num_layers)
        
    def get_trainable_params(self, phase='phase1'):
        """
        获取不同训练阶段的可训练参数
        """
        if phase == 'phase1':
            params = list(self.slot_attention_head.parameters())
            params += list(self.motion_decoder.parameters())
        elif phase == 'phase2':
            params = [p for p in self.parameters() if p.requires_grad]
        else:
            raise ValueError(f"Unknown phase: {phase}")
            
        return params
    
    def identify_foreground(self, masks, slot_flows, motion_info=None):
        """
        识别前景物体
        
        策略:
        1. 如果有全局光流，计算每个 slot 相对于全局的残差
        2. 残差大的是前景，残差小的是背景
        
        Args:
            masks: [B, K, H, W] 分割 masks
            slot_flows: [B, K, 2, H, W] 每个 slot 的光流
            motion_info: dict 包含全局光流等信息
            
        Returns:
            fg_mask: [B, 1, H, W] 前景 mask
            bg_idx: [B] 背景 slot 索引
        """
        B, K, H, W = masks.shape
        device = masks.device
        
        # 计算每个 slot 的运动幅度
        slot_motion_mag = self.motion_decoder.compute_motion_magnitude(slot_flows)  # [B, K]
        
        if motion_info and 'global_flow' in motion_info:
            # 计算相对于全局运动的残差
            global_flow = motion_info['global_flow']  # [B, 2, H_s, W_s]
            
            # 上采样到 mask 尺寸
            if global_flow.shape[-2:] != (H, W):
                global_flow = F.interpolate(global_flow, size=(H, W), mode='bilinear', align_corners=False)
            
            # 计算每个 slot 与全局光流的差异
            residual_mag = []
            for k in range(K):
                slot_flow_k = slot_flows[:, k]  # [B, 2, H_s, W_s]
                if slot_flow_k.shape[-2:] != (H, W):
                    slot_flow_k = F.interpolate(slot_flow_k, size=(H, W), mode='bilinear', align_corners=False)
                
                # 残差
                residual = slot_flow_k - global_flow
                residual_mag_k = torch.sqrt(residual[:, 0] ** 2 + residual[:, 1] ** 2 + 1e-8)
                
                # 用 mask 加权平均
                weighted_residual = (residual_mag_k * masks[:, k]).sum(dim=(-2, -1)) / (masks[:, k].sum(dim=(-2, -1)) + 1e-8)
                residual_mag.append(weighted_residual)
            
            residual_mag = torch.stack(residual_mag, dim=1)  # [B, K]
            
            # 背景 = 残差最小的 slot
            bg_idx = residual_mag.argmin(dim=1)  # [B]
            
            # 前景 = 残差大于阈值的 slots
            threshold = residual_mag.median(dim=1, keepdim=True)[0]
            fg_slots = residual_mag > threshold
            
        else:
            # 没有全局光流，用面积和运动幅度判断
            areas = masks.sum(dim=(-2, -1))  # [B, K]
            areas_norm = areas / areas.sum(dim=1, keepdim=True)
            motion_norm = slot_motion_mag / (slot_motion_mag.sum(dim=1, keepdim=True) + 1e-8)
            
            # 背景 = 面积大 + 运动小
            bg_score = areas_norm - motion_norm
            bg_idx = bg_score.argmax(dim=1)
            
            # 前景 = 非背景
            fg_slots = torch.ones(B, K, device=device, dtype=torch.bool)
            for b in range(B):
                fg_slots[b, bg_idx[b]] = False
        
        # 生成前景 mask
        fg_mask = torch.zeros(B, 1, H, W, device=device)
        for b in range(B):
            for k in range(K):
                if fg_slots[b, k]:
                    fg_mask[b, 0] += masks[b, k]
        
        fg_mask = torch.clamp(fg_mask, 0, 1)
        
        return fg_mask, bg_idx
