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
    仿射运动模型
    每个 slot 预测 6 个仿射参数: [a, b, tx, c, d, ty]
    flow(x, y) = [a*x + b*y + tx, c*x + d*y + ty]
    """
    
    def __init__(self, slot_dim: int = 64, hidden_dim: int = 128):
        super().__init__()
        
        self.motion_mlp = nn.Sequential(
            nn.Linear(slot_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 6)  # 6 个仿射参数
        )
        
        # 初始化为恒等变换
        self.motion_mlp[-1].weight.data.zero_()
        self.motion_mlp[-1].bias.data = torch.tensor([1., 0., 0., 0., 1., 0.])
        
    def forward(self, slots, coords):
        """
        Args:
            slots: [B, K, D] slot 特征
            coords: [B, 2, H, W] 像素坐标网格
            
        Returns:
            flows: [B, K, 2, H, W] 每个 slot 的运动场
        """
        B, K, D = slots.shape
        _, _, H, W = coords.shape
        
        # 预测仿射参数
        affine_params = self.motion_mlp(slots)  # [B, K, 6]
        
        # 解析参数
        a = affine_params[..., 0:1]  # [B, K, 1]
        b = affine_params[..., 1:2]
        tx = affine_params[..., 2:3]
        c = affine_params[..., 3:4]
        d = affine_params[..., 4:5]
        ty = affine_params[..., 5:6]
        
        # 坐标网格
        x = coords[:, 0:1]  # [B, 1, H, W]
        y = coords[:, 1:2]  # [B, 1, H, W]
        
        # 计算每个 slot 的光流
        # flow_x = (a-1)*x + b*y + tx  (减去恒等变换)
        # flow_y = c*x + (d-1)*y + ty
        a = a.unsqueeze(-1).unsqueeze(-1)  # [B, K, 1, 1, 1]
        b = b.unsqueeze(-1).unsqueeze(-1)
        tx = tx.unsqueeze(-1).unsqueeze(-1)
        c = c.unsqueeze(-1).unsqueeze(-1)
        d = d.unsqueeze(-1).unsqueeze(-1)
        ty = ty.unsqueeze(-1).unsqueeze(-1)
        
        x = x.unsqueeze(1)  # [B, 1, 1, H, W]
        y = y.unsqueeze(1)
        
        flow_x = (a - 1) * x + b * y + tx  # [B, K, 1, H, W]
        flow_y = c * x + (d - 1) * y + ty
        
        flows = torch.cat([flow_x, flow_y], dim=2)  # [B, K, 2, H, W]
        flows = flows.squeeze(3) if flows.dim() == 6 else flows
        
        return flows, affine_params


class QuadraticMotionModel(nn.Module):
    """
    二次运动模型 (更灵活，可以建模旋转和缩放)
    每个 slot 预测 12 个参数
    """
    
    def __init__(self, slot_dim: int = 64, hidden_dim: int = 128):
        super().__init__()
        
        self.motion_mlp = nn.Sequential(
            nn.Linear(slot_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 12)  # 12 个二次参数
        )
        
        # 初始化为零运动
        self.motion_mlp[-1].weight.data.zero_()
        self.motion_mlp[-1].bias.data.zero_()
        
    def forward(self, slots, coords):
        """
        Args:
            slots: [B, K, D] slot 特征
            coords: [B, 2, H, W] 像素坐标网格
            
        Returns:
            flows: [B, K, 2, H, W] 每个 slot 的运动场
        """
        B, K, D = slots.shape
        _, _, H, W = coords.shape
        
        # 预测参数
        params = self.motion_mlp(slots)  # [B, K, 12]
        
        # 坐标
        x = coords[:, 0:1].unsqueeze(1)  # [B, 1, 1, H, W]
        y = coords[:, 1:2].unsqueeze(1)
        
        # 归一化坐标到 [-1, 1]
        x_norm = 2 * x / W - 1
        y_norm = 2 * y / H - 1
        
        # 解析参数 [B, K, 12] -> 各项系数
        p = params.view(B, K, 12, 1, 1)
        
        # flow_x = p0 + p1*x + p2*y + p3*x^2 + p4*y^2 + p5*xy
        # flow_y = p6 + p7*x + p8*y + p9*x^2 + p10*y^2 + p11*xy
        flow_x = (p[:,:,0] + p[:,:,1]*x_norm + p[:,:,2]*y_norm + 
                  p[:,:,3]*x_norm**2 + p[:,:,4]*y_norm**2 + p[:,:,5]*x_norm*y_norm)
        flow_y = (p[:,:,6] + p[:,:,7]*x_norm + p[:,:,8]*y_norm + 
                  p[:,:,9]*x_norm**2 + p[:,:,10]*y_norm**2 + p[:,:,11]*x_norm*y_norm)
        
        # 缩放回像素空间
        flow_x = flow_x * W / 2
        flow_y = flow_y * H / 2
        
        flows = torch.cat([flow_x, flow_y], dim=2)  # [B, K, 2, H, W]
        
        return flows, params


class MotionDecoder(nn.Module):
    """
    完整的 Motion Decoder
    结合 Slot 特征和 Masks 重建光流场
    """
    
    def __init__(
        self,
        slot_dim: int = 64,
        hidden_dim: int = 128,
        motion_model: str = 'affine'  # 'affine' or 'quadratic'
    ):
        super().__init__()
        
        self.motion_model_type = motion_model
        
        if motion_model == 'affine':
            self.motion_model = AffineMotionModel(slot_dim, hidden_dim)
        elif motion_model == 'quadratic':
            self.motion_model = QuadraticMotionModel(slot_dim, hidden_dim)
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
    
    def forward(self, slots, masks, target_size):
        """
        Args:
            slots: [B, K, D] slot 特征
            masks: [B, K, H, W] 分割 masks (softmax 归一化)
            target_size: (H, W) 目标光流尺寸
            
        Returns:
            flow_recon: [B, 2, H, W] 重建的光流
            slot_flows: [B, K, 2, H, W] 每个 slot 的运动场
            motion_params: [B, K, P] 运动参数
        """
        B, K, D = slots.shape
        H, W = target_size
        device = slots.device
        
        # 创建坐标网格
        coords = self.create_coord_grid(B, H, W, device)
        
        # 预测每个 slot 的运动场
        slot_flows, motion_params = self.motion_model(slots, coords)  # [B, K, 2, H, W]
        
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
            # 仿射模型: 运动幅度 ≈ |tx| + |ty| + |a-1| + |d-1| + |b| + |c|
            a, b, tx, c, d, ty = motion_params.split(1, dim=-1)
            magnitude = (tx.abs() + ty.abs() + 
                        (a - 1).abs() + (d - 1).abs() + 
                        b.abs() + c.abs()).squeeze(-1)
        else:
            # 二次模型: 使用所有参数的 L2 范数
            magnitude = motion_params.norm(dim=-1)
            
        return magnitude  # [B, K]
