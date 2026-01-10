"""
Motion Decoder - 重新设计的复杂运动模型
支持非对齐视角下的变化检测

核心改进:
1. 密集光流预测: 每个 Slot 预测完整的密集光流场，而非简单参数
2. 多尺度解码: 使用 CNN Decoder 逐步上采样，捕获不同尺度的运动
3. 残差学习: 预测相对于全局运动的残差，更容易学习
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import math


class ConvBlock(nn.Module):
    """基础卷积块"""
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding),
            nn.GroupNorm(8, out_ch),
            nn.GELU()
        )
    
    def forward(self, x):
        return self.conv(x)


class UpBlock(nn.Module):
    """上采样块"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_ch, out_ch, 3, 1, 1),
            nn.GroupNorm(8, out_ch),
            nn.GELU()
        )
    
    def forward(self, x):
        return self.up(x)


class DenseFlowDecoder(nn.Module):
    """
    密集光流解码器
    从 Slot 特征解码出完整的密集光流场
    """
    
    def __init__(self, slot_dim: int = 64, hidden_dim: int = 128):
        super().__init__()
        
        self.slot_dim = slot_dim
        
        # Slot 特征投影
        self.slot_proj = nn.Sequential(
            nn.Linear(slot_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 位置编码生成器
        self.pos_encoder = nn.Sequential(
            nn.Linear(2, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, hidden_dim)
        )
        
        # 密集光流解码器 (类似 NeRF 的 MLP)
        self.flow_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 2)  # 输出 (flow_x, flow_y)
        )
        
        # 初始化最后一层为小值
        nn.init.zeros_(self.flow_mlp[-1].bias)
        nn.init.normal_(self.flow_mlp[-1].weight, std=0.01)

    def forward(self, slots, H, W, full_size):
        """
        Args:
            slots: [B, K, D] slot 特征
            H, W: 特征图尺寸
            full_size: (H_full, W_full) 原始图像尺寸
            
        Returns:
            flows: [B, K, 2, H, W] 每个 slot 的密集光流场
        """
        B, K, D = slots.shape
        H_full, W_full = full_size
        device = slots.device
        
        # 投影 slot 特征
        slot_feat = self.slot_proj(slots)  # [B, K, hidden_dim]
        
        # 创建归一化坐标网格
        y = torch.linspace(-1, 1, H, device=device)
        x = torch.linspace(-1, 1, W, device=device)
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        coords = torch.stack([xx, yy], dim=-1)  # [H, W, 2]
        coords = coords.view(1, 1, H * W, 2).expand(B, K, -1, -1)  # [B, K, H*W, 2]
        
        # 位置编码
        pos_enc = self.pos_encoder(coords)  # [B, K, H*W, hidden_dim]
        
        # 扩展 slot 特征到每个位置
        slot_feat_expanded = slot_feat.unsqueeze(2).expand(-1, -1, H * W, -1)  # [B, K, H*W, hidden_dim]
        
        # 拼接 slot 特征和位置编码
        combined = torch.cat([slot_feat_expanded, pos_enc], dim=-1)  # [B, K, H*W, hidden_dim*2]
        
        # 预测光流
        flow_pred = self.flow_mlp(combined)  # [B, K, H*W, 2]
        
        # 缩放到像素空间
        flow_pred = flow_pred * torch.tensor([W_full / 2, H_full / 2], device=device)
        
        # 重塑为空间形式
        flows = flow_pred.view(B, K, H, W, 2).permute(0, 1, 4, 2, 3)  # [B, K, 2, H, W]
        
        return flows


class CNNFlowDecoder(nn.Module):
    """
    CNN 密集光流解码器
    使用卷积网络从 Slot 特征解码光流，更高效
    """
    
    def __init__(self, slot_dim: int = 64, hidden_dim: int = 128, output_scale: int = 8):
        super().__init__()
        
        self.slot_dim = slot_dim
        self.output_scale = output_scale
        
        # Slot 到空间特征的投影
        self.slot_to_spatial = nn.Sequential(
            nn.Linear(slot_dim, hidden_dim * 4 * 4),
            nn.GELU()
        )
        
        # CNN 解码器
        self.decoder = nn.Sequential(
            # 4x4 -> 8x8
            UpBlock(hidden_dim, hidden_dim),
            ConvBlock(hidden_dim, hidden_dim),
            # 8x8 -> 16x16
            UpBlock(hidden_dim, hidden_dim // 2),
            ConvBlock(hidden_dim // 2, hidden_dim // 2),
            # 16x16 -> 32x32
            UpBlock(hidden_dim // 2, hidden_dim // 4),
            ConvBlock(hidden_dim // 4, hidden_dim // 4),
        )
        
        # 光流预测头
        self.flow_head = nn.Sequential(
            nn.Conv2d(hidden_dim // 4, hidden_dim // 4, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(hidden_dim // 4, 2, 3, 1, 1)
        )
        
        # 初始化 - 使用较大的标准差
        nn.init.zeros_(self.flow_head[-1].bias)
        nn.init.normal_(self.flow_head[-1].weight, std=0.1)
        
        # 可学习的缩放因子
        self.flow_scale = nn.Parameter(torch.tensor(100.0))
    
    def forward(self, slots, H, W, full_size):
        """
        Args:
            slots: [B, K, D] slot 特征
            H, W: 目标特征图尺寸
            full_size: (H_full, W_full) 原始图像尺寸
            
        Returns:
            flows: [B, K, 2, H, W] 每个 slot 的密集光流场
        """
        B, K, D = slots.shape
        H_full, W_full = full_size
        device = slots.device
        
        flows_list = []
        
        for k in range(K):
            slot_k = slots[:, k]  # [B, D]
            
            # 投影到空间特征
            spatial = self.slot_to_spatial(slot_k)  # [B, hidden_dim * 16]
            spatial = spatial.view(B, -1, 4, 4)  # [B, hidden_dim, 4, 4]
            
            # CNN 解码
            feat = self.decoder(spatial)  # [B, hidden_dim//4, 32, 32]
            
            # 上采样到目标尺寸
            feat = F.interpolate(feat, size=(H, W), mode='bilinear', align_corners=False)
            
            # 预测光流
            flow_k = self.flow_head(feat)  # [B, 2, H, W]
            
            # 使用可学习的缩放因子
            flow_k = flow_k * self.flow_scale
            
            flows_list.append(flow_k)
        
        flows = torch.stack(flows_list, dim=1)  # [B, K, 2, H, W]
        
        return flows


class HybridMotionDecoder(nn.Module):
    """
    混合运动解码器 - 最强大的版本
    
    结合:
    1. 全局单应性估计 (处理视角变化)
    2. 局部残差光流 (处理独立运动物体)
    
    这样可以:
    - 全局单应性补偿相机运动/视角变化
    - 残差光流捕获前景物体的独立运动
    """
    
    def __init__(self, slot_dim: int = 64, hidden_dim: int = 128, context_dim: int = 256):
        super().__init__()
        
        self.slot_dim = slot_dim
        self.context_dim = context_dim
        
        # 全局运动估计器 (从 context 特征估计)
        self.global_motion_encoder = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(context_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 8)  # 8 参数的单应性 (去掉 scale)
        )
        
        # 使用正常初始化
        nn.init.xavier_uniform_(self.global_motion_encoder[-1].weight)
        nn.init.zeros_(self.global_motion_encoder[-1].bias)
        
        # 可学习的单应性参数缩放因子
        self.homography_scale = nn.Parameter(torch.tensor(0.1))
        
        # 每个 Slot 的残差光流解码器
        self.slot_proj = nn.Sequential(
            nn.Linear(slot_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU()
        )
        
        # 残差光流 CNN - 输出经过 tanh 限制范围
        self.residual_decoder = nn.Sequential(
            nn.Conv2d(hidden_dim + 2, hidden_dim, 3, 1, 1),  # +2 for position
            nn.GroupNorm(8, hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1),
            nn.GroupNorm(8, hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1),
            nn.GroupNorm(8, hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, 2, 3, 1, 1)
            # 不使用 tanh，让网络自由学习
        )
        
        # 使用正常初始化
        nn.init.xavier_uniform_(self.residual_decoder[-1].weight)
        nn.init.zeros_(self.residual_decoder[-1].bias)
        
        # 可学习的缩放因子，初始化为合理值
        self.residual_scale = nn.Parameter(torch.tensor(30.0))
    
    def compute_homography_flow(self, H_params, coords, H_full, W_full):
        """
        从单应性参数计算光流
        
        H_params: [B, 8] 单应性参数 (h11-1, h12, h13, h21, h22-1, h23, h31, h32)
        coords: [B, 2, H, W] 归一化坐标
        """
        B = H_params.shape[0]
        _, _, H, W = coords.shape
        device = H_params.device
        
        # 构建单应性矩阵
        # H = [[1+h11, h12, h13], [h21, 1+h22, h23], [h31, h32, 1]]
        h11 = H_params[:, 0].view(B, 1, 1)
        h12 = H_params[:, 1].view(B, 1, 1)
        h13 = H_params[:, 2].view(B, 1, 1)
        h21 = H_params[:, 3].view(B, 1, 1)
        h22 = H_params[:, 4].view(B, 1, 1)
        h23 = H_params[:, 5].view(B, 1, 1)
        h31 = H_params[:, 6].view(B, 1, 1)
        h32 = H_params[:, 7].view(B, 1, 1)
        
        # 归一化坐标
        x = coords[:, 0]  # [B, H, W]
        y = coords[:, 1]
        
        # 单应性变换: x' = (h11*x + h12*y + h13) / (h31*x + h32*y + 1)
        denom = h31 * x + h32 * y + 1.0
        denom = torch.clamp(denom, min=1e-6)  # 防止除零
        
        x_new = ((1 + h11) * x + h12 * y + h13) / denom
        y_new = (h21 * x + (1 + h22) * y + h23) / denom
        
        # 光流 = 新坐标 - 原坐标 (归一化空间)
        flow_x_norm = x_new - x
        flow_y_norm = y_new - y
        
        # 转换到像素空间
        flow_x = flow_x_norm * (W_full / 2)
        flow_y = flow_y_norm * (H_full / 2)
        
        return torch.stack([flow_x, flow_y], dim=1)  # [B, 2, H, W]
    
    def forward(self, slots, masks, context_feat, target_size, full_size):
        """
        Args:
            slots: [B, K, D] slot 特征
            masks: [B, K, H, W] 分割 masks
            context_feat: [B, C, H_ctx, W_ctx] 上下文特征
            target_size: (H, W) 目标光流尺寸
            full_size: (H_full, W_full) 原始图像尺寸
            
        Returns:
            flow_recon: [B, 2, H, W] 重建的光流
            slot_flows: [B, K, 2, H, W] 每个 slot 的光流
            global_flow: [B, 2, H, W] 全局光流
            motion_info: dict 包含运动参数
        """
        B, K, D = slots.shape
        H, W = target_size
        H_full, W_full = full_size
        device = slots.device
        
        # 1. 估计全局单应性 (输出已经经过 tanh，范围 [-1, 1])
        H_params = self.global_motion_encoder(context_feat)  # [B, 8]
        # 应用缩放因子
        H_params = H_params * self.homography_scale
        
        # 创建归一化坐标网格
        y = torch.linspace(-1, 1, H, device=device)
        x = torch.linspace(-1, 1, W, device=device)
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        coords = torch.stack([xx, yy], dim=0).unsqueeze(0).expand(B, -1, -1, -1)  # [B, 2, H, W]
        
        # 计算全局光流
        global_flow = self.compute_homography_flow(H_params, coords, H_full, W_full)  # [B, 2, H, W]
        
        # 2. 计算每个 Slot 的残差光流
        slot_feat = self.slot_proj(slots)  # [B, K, hidden_dim]
        
        slot_flows_list = []
        for k in range(K):
            # 广播 slot 特征到空间
            feat_k = slot_feat[:, k].unsqueeze(-1).unsqueeze(-1)  # [B, hidden_dim, 1, 1]
            feat_k = feat_k.expand(-1, -1, H, W)  # [B, hidden_dim, H, W]
            
            # 拼接位置信息
            feat_with_pos = torch.cat([feat_k, coords], dim=1)  # [B, hidden_dim+2, H, W]
            
            # 预测残差光流
            residual_flow = self.residual_decoder(feat_with_pos)  # [B, 2, H, W]
            # 使用可学习的缩放因子
            residual_flow = residual_flow * self.residual_scale
            
            # 总光流 = 全局光流 + 残差光流
            total_flow_k = global_flow + residual_flow
            slot_flows_list.append(total_flow_k)
        
        slot_flows = torch.stack(slot_flows_list, dim=1)  # [B, K, 2, H, W]
        
        # 3. 调整 masks 尺寸
        if masks.shape[-2:] != (H, W):
            masks = F.interpolate(masks, size=(H, W), mode='bilinear', align_corners=False)
            masks = F.softmax(masks, dim=1)
        
        # 4. 加权组合
        masks_expanded = masks.unsqueeze(2)  # [B, K, 1, H, W]
        flow_recon = (masks_expanded * slot_flows).sum(dim=1)  # [B, 2, H, W]
        
        motion_info = {
            'homography_params': H_params,
            'global_flow': global_flow
        }
        
        return flow_recon, slot_flows, global_flow, motion_info


class MotionDecoder(nn.Module):
    """
    统一的 Motion Decoder 接口
    支持多种运动模型
    """
    
    def __init__(
        self,
        slot_dim: int = 64,
        hidden_dim: int = 128,
        context_dim: int = 256,
        motion_model: str = 'hybrid'  # 'dense', 'cnn', 'hybrid'
    ):
        super().__init__()
        
        self.motion_model_type = motion_model
        self.context_dim = context_dim
        
        if motion_model == 'dense':
            self.motion_model = DenseFlowDecoder(slot_dim, hidden_dim)
        elif motion_model == 'cnn':
            self.motion_model = CNNFlowDecoder(slot_dim, hidden_dim)
        elif motion_model == 'hybrid':
            self.motion_model = HybridMotionDecoder(slot_dim, hidden_dim, context_dim)
        else:
            raise ValueError(f"Unknown motion model: {motion_model}")
    
    def forward(self, slots, masks, target_size, full_size=None, context_feat=None):
        """
        Args:
            slots: [B, K, D] slot 特征
            masks: [B, K, H_m, W_m] 分割 masks
            target_size: (H, W) 目标光流尺寸
            full_size: (H_full, W_full) 原始图像尺寸
            context_feat: [B, C, H_ctx, W_ctx] 上下文特征 (hybrid 模式需要)
            
        Returns:
            flow_recon: [B, 2, H, W] 重建的光流
            slot_flows: [B, K, 2, H, W] 每个 slot 的光流
            motion_info: dict 运动信息
        """
        B, K, D = slots.shape
        H, W = target_size
        
        if full_size is None:
            full_size = (H * 8, W * 8)
        
        if self.motion_model_type == 'hybrid':
            if context_feat is None:
                raise ValueError("Hybrid model requires context_feat")
            flow_recon, slot_flows, global_flow, motion_info = self.motion_model(
                slots, masks, context_feat, target_size, full_size
            )
        else:
            # dense 或 cnn 模式
            slot_flows = self.motion_model(slots, H, W, full_size)  # [B, K, 2, H, W]
            
            # 调整 masks 尺寸
            if masks.shape[-2:] != (H, W):
                masks = F.interpolate(masks, size=(H, W), mode='bilinear', align_corners=False)
                masks = F.softmax(masks, dim=1)
            
            # 加权组合
            masks_expanded = masks.unsqueeze(2)  # [B, K, 1, H, W]
            flow_recon = (masks_expanded * slot_flows).sum(dim=1)  # [B, 2, H, W]
            
            motion_info = {}
        
        return flow_recon, slot_flows, motion_info
    
    def compute_motion_magnitude(self, slot_flows):
        """
        计算每个 slot 的运动幅度
        
        Args:
            slot_flows: [B, K, 2, H, W]
            
        Returns:
            magnitude: [B, K] 每个 slot 的平均运动幅度
        """
        # 计算每个位置的光流幅度
        flow_mag = torch.sqrt(slot_flows[:, :, 0] ** 2 + slot_flows[:, :, 1] ** 2 + 1e-8)  # [B, K, H, W]
        
        # 平均
        magnitude = flow_mag.mean(dim=(-2, -1))  # [B, K]
        
        return magnitude
