"""
Motion Decoder - 从 Slot 特征解码运动参数，用于重建光流
核心思想: 每个 slot 代表一个具有一致运动模式的区域
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


class AffineMotionModel(nn.Module):
    """
    仿射运动模型 - 简化版本
    每个 slot 直接预测光流的仿射参数
    flow(x, y) = [a0 + a1*x_norm + a2*y_norm, b0 + b1*x_norm + b2*y_norm]
    其中 x_norm, y_norm 是归一化到 [-1, 1] 的坐标
    """
    
    def __init__(self, slot_dim: int = 64, hidden_dim: int = 128):
        super().__init__()
        
        self.motion_mlp = nn.Sequential(
            nn.Linear(slot_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 6)  # [a0, a1, a2, b0, b1, b2]
        )
        
        # 使用 Xavier 初始化
        for m in self.motion_mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        
    def forward(self, slots, coords, image_size):
        """
        Args:
            slots: [B, K, D] slot 特征
            coords: [B, 2, H, W] 像素坐标网格 (x, y)
            image_size: (H_full, W_full) 原始图像尺寸，用于缩放
            
        Returns:
            flows: [B, K, 2, H, W] 每个 slot 的运动场 (像素单位)
            affine_params: [B, K, 6] 仿射参数
        """
        B, K, D = slots.shape
        _, _, H, W = coords.shape
        H_full, W_full = image_size
        
        # 预测仿射参数
        affine_params = self.motion_mlp(slots)  # [B, K, 6]
        
        # 归一化坐标到 [-1, 1]
        x = coords[:, 0:1]  # [B, 1, H, W]
        y = coords[:, 1:2]
        x_norm = 2.0 * x / max(W - 1, 1) - 1.0  # [-1, 1]
        y_norm = 2.0 * y / max(H - 1, 1) - 1.0
        
        # 解析参数 [B, K, 6] - 乘以缩放因子让输出在合理范围
        params_scaled = affine_params * 30.0  # 缩放到合理范围
        
        a0 = params_scaled[:, :, 0].view(B, K, 1, 1)  # [B, K, 1, 1]
        a1 = params_scaled[:, :, 1].view(B, K, 1, 1)
        a2 = params_scaled[:, :, 2].view(B, K, 1, 1)
        b0 = params_scaled[:, :, 3].view(B, K, 1, 1)
        b1 = params_scaled[:, :, 4].view(B, K, 1, 1)
        b2 = params_scaled[:, :, 5].view(B, K, 1, 1)
        
        # x_norm, y_norm: [B, 1, H, W] -> 广播到 [B, K, H, W]
        x_norm = x_norm.squeeze(1)  # [B, H, W]
        y_norm = y_norm.squeeze(1)  # [B, H, W]
        
        # 计算光流 - 直接在像素空间计算
        # x_norm, y_norm 在 [-1, 1]，乘以参数后得到像素位移
        flow_x = a0 + a1 * x_norm.unsqueeze(1) + a2 * y_norm.unsqueeze(1)  # [B, K, H, W]
        flow_y = b0 + b1 * x_norm.unsqueeze(1) + b2 * y_norm.unsqueeze(1)  # [B, K, H, W]
        
        # 堆叠成 [B, K, 2, H, W]
        flows = torch.stack([flow_x, flow_y], dim=2)  # [B, K, 2, H, W]
        
        return flows, affine_params


class TranslationMotionModel(nn.Module):
    """
    纯平移运动模型 - 最简单的模型
    每个 slot 只预测一个平移向量 (tx, ty)
    """
    
    def __init__(self, slot_dim: int = 64, hidden_dim: int = 128):
        super().__init__()
        
        self.motion_mlp = nn.Sequential(
            nn.Linear(slot_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 2)  # [tx, ty] 直接输出像素单位的位移
        )
        
        # 使用 Xavier 初始化，让输出有合理的初始范围
        for m in self.motion_mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        
    def forward(self, slots, coords, image_size):
        """
        Args:
            slots: [B, K, D] slot 特征
            coords: [B, 2, H, W] 像素坐标网格
            image_size: (H_full, W_full) 原始图像尺寸
            
        Returns:
            flows: [B, K, 2, H, W] 每个 slot 的运动场
            motion_params: [B, K, 2] 平移参数
        """
        B, K, D = slots.shape
        _, _, H, W = coords.shape
        H_full, W_full = image_size
        
        # 预测平移参数 - 直接输出像素单位的位移
        # 乘以一个缩放因子让初始输出在合理范围
        translation = self.motion_mlp(slots) * 50.0  # [B, K, 2], 缩放到 ~[-50, 50] 像素
        
        tx = translation[:, :, 0]  # [B, K]
        ty = translation[:, :, 1]  # [B, K]
        
        # 广播到空间维度 [B, K] -> [B, K, H, W]
        flow_x = tx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)  # [B, K, H, W]
        flow_y = ty.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)  # [B, K, H, W]
        
        # 堆叠成 [B, K, 2, H, W]
        flows = torch.stack([flow_x, flow_y], dim=2)  # [B, K, 2, H, W]
        
        return flows, translation


class MotionDecoder(nn.Module):
    """
    完整的 Motion Decoder
    结合 Slot 特征和 Masks 重建光流场
    """
    
    def __init__(
        self,
        slot_dim: int = 64,
        hidden_dim: int = 128,
        motion_model: str = 'affine'  # 'affine' or 'translation'
    ):
        super().__init__()
        
        self.motion_model_type = motion_model
        
        if motion_model == 'affine':
            self.motion_model = AffineMotionModel(slot_dim, hidden_dim)
        elif motion_model == 'translation':
            self.motion_model = TranslationMotionModel(slot_dim, hidden_dim)
        else:
            raise ValueError(f"Unknown motion model: {motion_model}")
            
    def create_coord_grid(self, B, H, W, device):
        """创建像素坐标网格"""
        y = torch.arange(H, device=device).float()
        x = torch.arange(W, device=device).float()
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        coords = torch.stack([xx, yy], dim=0)  # [2, H, W]
        coords = coords.unsqueeze(0).repeat(B, 1, 1, 1)  # [B, 2, H, W]
        return coords
    
    def forward(self, slots, masks, target_size, full_size=None):
        """
        Args:
            slots: [B, K, D] slot 特征
            masks: [B, K, H, W] 分割 masks (softmax 归一化)
            target_size: (H, W) 目标光流尺寸
            full_size: (H_full, W_full) 原始图像尺寸，用于缩放光流
            
        Returns:
            flow_recon: [B, 2, H, W] 重建的光流
            slot_flows: [B, K, 2, H, W] 每个 slot 的运动场
            motion_params: [B, K, P] 运动参数
        """
        B, K, D = slots.shape
        H, W = target_size
        device = slots.device
        
        # 如果没有指定 full_size，假设是 8x 下采样
        if full_size is None:
            full_size = (H * 8, W * 8)
        
        # 创建坐标网格
        coords = self.create_coord_grid(B, H, W, device)
        
        # 预测每个 slot 的运动场
        slot_flows, motion_params = self.motion_model(slots, coords, full_size)  # [B, K, 2, H, W]
        
        # 调整 masks 到目标尺寸
        if masks.shape[-2:] != (H, W):
            masks = F.interpolate(masks, size=(H, W), mode='bilinear', align_corners=False)
            masks = F.softmax(masks, dim=1)  # 重新归一化
        
        # 加权组合: flow = sum_k(mask_k * flow_k)
        masks_expanded = masks.unsqueeze(2)  # [B, K, 1, H, W]
        flow_recon = (masks_expanded * slot_flows).sum(dim=1)  # [B, 2, H, W]
        
        return flow_recon, slot_flows, motion_params
    
    def compute_motion_magnitude(self, motion_params):
        """
        计算每个 slot 的运动幅度
        用于识别背景 (运动最小的 slot)
        """
        if self.motion_model_type == 'affine':
            # 仿射模型: 使用所有参数的 L2 范数
            magnitude = motion_params.abs().sum(dim=-1)
        elif self.motion_model_type == 'translation':
            # 平移模型: 使用平移向量的 L2 范数
            magnitude = motion_params.norm(dim=-1)
        else:
            magnitude = motion_params.norm(dim=-1)
            
        return magnitude  # [B, K]
