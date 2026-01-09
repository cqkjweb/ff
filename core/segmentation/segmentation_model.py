"""
Motion Segmentation Model - 完整的多任务模型
结合 FlowFormer + Slot Attention + Motion Decoder
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
    3. Motion Decoder 从 slots 解码运动参数
    4. 使用 masks 和运动参数重建光流
    """
    
    def __init__(
        self,
        flowformer_model,
        num_slots: int = 7,
        slot_dim: int = 64,
        hidden_dim: int = 128,
        num_iterations: int = 3,
        motion_model: str = 'affine',
        feature_scale: str = '1/8'
    ):
        """
        Args:
            flowformer_model: 预训练的 FlowFormer 模型
            num_slots: slot 数量 (K)
            slot_dim: slot 特征维度
            hidden_dim: MLP 隐藏层维度
            num_iterations: Slot Attention 迭代次数
            motion_model: 运动模型类型 ('affine' or 'quadratic')
            feature_scale: 特征图分辨率
        """
        super().__init__()
        
        self.num_slots = num_slots
        
        # FlowFormer Wrapper
        self.flowformer_wrapper = FlowFormerWrapper(
            flowformer_model, 
            feature_scale=feature_scale
        )
        
        # Slot Attention Head
        input_dim = self.flowformer_wrapper.context_dim
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
                - flow_pred: FlowFormer 预测的光流 (如果 return_flow=True)
                - flow_recon: 重建的光流
                - masks: [B, K, H, W] 分割 masks
                - slots: [B, K, D] slot 特征
                - slot_flows: [B, K, 2, H, W] 每个 slot 的运动场
                - motion_params: [B, K, P] 运动参数
        """
        B, _, H, W = image1.shape
        
        # 1. 提取 FlowFormer 特征
        ff_output = self.flowformer_wrapper(image1, image2)
        context_features = ff_output['context_features']
        feature_size = ff_output['feature_size']
        
        # 2. Slot Attention
        slots, masks, attn_masks = self.slot_attention_head(
            context_features, feature_size
        )
        
        # 3. Motion Decoder - 重建光流
        # 目标尺寸: 1/8 分辨率 (与 FlowFormer 内部一致)
        H8, W8 = H // 8, W // 8
        flow_recon, slot_flows, motion_params = self.motion_decoder(
            slots, attn_masks, target_size=(H8, W8)
        )
        
        # 上采样重建光流到原始分辨率
        flow_recon_full = F.interpolate(
            flow_recon, size=(H, W), mode='bilinear', align_corners=False
        ) * 8  # 缩放因子
        
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
            'motion_params': motion_params
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
        
        Args:
            phase: 'phase1' (只训练 Seg Head) 或 'phase2' (联合训练)
        """
        if phase == 'phase1':
            # 只返回 Slot Attention 和 Motion Decoder 的参数
            params = list(self.slot_attention_head.parameters())
            params += list(self.motion_decoder.parameters())
        elif phase == 'phase2':
            # 返回所有可训练参数
            params = [p for p in self.parameters() if p.requires_grad]
        else:
            raise ValueError(f"Unknown phase: {phase}")
            
        return params
    
    def identify_background(self, masks, motion_params):
        """
        识别背景 slot (面积最大且运动最小)
        
        Args:
            masks: [B, K, H, W] 分割 masks
            motion_params: [B, K, P] 运动参数
            
        Returns:
            bg_idx: [B] 每个样本的背景 slot 索引
            fg_mask: [B, 1, H, W] 前景 mask
        """
        B, K, H, W = masks.shape
        
        # 计算每个 slot 的面积
        areas = masks.sum(dim=(-2, -1))  # [B, K]
        
        # 计算运动幅度
        motion_mag = self.motion_decoder.compute_motion_magnitude(motion_params)  # [B, K]
        
        # 背景得分 = 面积大 + 运动小
        # 归一化
        areas_norm = areas / areas.sum(dim=1, keepdim=True)
        motion_norm = motion_mag / (motion_mag.sum(dim=1, keepdim=True) + 1e-8)
        
        bg_score = areas_norm - motion_norm  # 面积大、运动小的得分高
        bg_idx = bg_score.argmax(dim=1)  # [B]
        
        # 生成前景 mask
        bg_mask = torch.zeros_like(masks[:, 0:1])  # [B, 1, H, W]
        for b in range(B):
            bg_mask[b, 0] = masks[b, bg_idx[b]]
        fg_mask = 1 - bg_mask
        
        return bg_idx, fg_mask
