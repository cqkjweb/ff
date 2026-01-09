"""
Slot Attention Head - 将特征图映射到若干个 Slots，再解码回 Masks
参考: Locatello et al. "Object-Centric Learning with Slot Attention" (NeurIPS 2020)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import math


class SlotAttention(nn.Module):
    """
    Slot Attention 模块
    将输入特征聚类到 K 个 slots，每个 slot 代表一个物体/区域
    """
    
    def __init__(
        self,
        num_slots: int,
        input_dim: int,
        slot_dim: int = 64,
        hidden_dim: int = 128,
        num_iterations: int = 3,
        epsilon: float = 1e-8
    ):
        """
        Args:
            num_slots: slot 数量 (K)，代表最大物体数
            input_dim: 输入特征维度
            slot_dim: slot 特征维度
            hidden_dim: MLP 隐藏层维度
            num_iterations: 迭代次数
            epsilon: 数值稳定性
        """
        super().__init__()
        
        self.num_slots = num_slots
        self.num_iterations = num_iterations
        self.epsilon = epsilon
        self.slot_dim = slot_dim
        
        # Slot 初始化参数 (可学习的均值和方差)
        self.slots_mu = nn.Parameter(torch.randn(1, 1, slot_dim))
        self.slots_log_sigma = nn.Parameter(torch.zeros(1, 1, slot_dim))
        nn.init.xavier_uniform_(self.slots_mu)
        
        # 输入特征投影
        self.project_input = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, slot_dim)
        )
        
        # Q, K, V 投影
        self.to_q = nn.Linear(slot_dim, slot_dim, bias=False)
        self.to_k = nn.Linear(slot_dim, slot_dim, bias=False)
        self.to_v = nn.Linear(slot_dim, slot_dim, bias=False)
        
        # GRU 更新
        self.gru = nn.GRUCell(slot_dim, slot_dim)
        
        # MLP 残差更新
        self.mlp = nn.Sequential(
            nn.Linear(slot_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, slot_dim)
        )
        
        # Layer Norms
        self.norm_input = nn.LayerNorm(slot_dim)
        self.norm_slots = nn.LayerNorm(slot_dim)
        self.norm_mlp = nn.LayerNorm(slot_dim)
        
        self.scale = slot_dim ** -0.5

    def _init_slots(self, batch_size, device):
        """初始化 slots，使用可学习的高斯分布采样"""
        mu = self.slots_mu.expand(batch_size, self.num_slots, -1)
        sigma = self.slots_log_sigma.exp().expand(batch_size, self.num_slots, -1)
        slots = mu + sigma * torch.randn_like(mu)
        return slots
    
    def forward(self, inputs, num_slots=None):
        """
        Args:
            inputs: [B, N, D] 输入特征 (N = H*W)
            num_slots: 可选，覆盖默认 slot 数量
            
        Returns:
            slots: [B, K, slot_dim] 聚类后的 slot 特征
            attn_weights: [B, K, N] 注意力权重 (soft masks)
        """
        B, N, D = inputs.shape
        K = num_slots if num_slots is not None else self.num_slots
        
        # 投影输入特征
        inputs = self.project_input(inputs)  # [B, N, slot_dim]
        
        # 初始化 slots
        slots = self._init_slots(B, inputs.device)  # [B, K, slot_dim]
        
        # 计算 K, V (输入特征不变)
        inputs_normed = self.norm_input(inputs)
        k = self.to_k(inputs_normed)  # [B, N, slot_dim]
        v = self.to_v(inputs_normed)  # [B, N, slot_dim]
        
        # 迭代更新 slots
        for _ in range(self.num_iterations):
            slots_prev = slots
            slots = self.norm_slots(slots)
            
            # 计算 Q
            q = self.to_q(slots)  # [B, K, slot_dim]
            
            # 计算注意力: softmax over slots (竞争机制)
            # attn[b, k, n] = softmax_k(q[b,k] · k[b,n])
            attn_logits = torch.einsum('bkd,bnd->bkn', q, k) * self.scale
            attn = F.softmax(attn_logits, dim=1)  # [B, K, N], softmax over K
            
            # 归一化注意力权重
            attn_weights = attn / (attn.sum(dim=-1, keepdim=True) + self.epsilon)
            
            # 加权聚合
            updates = torch.einsum('bkn,bnd->bkd', attn_weights, v)  # [B, K, slot_dim]
            
            # GRU 更新
            slots = self.gru(
                updates.reshape(-1, self.slot_dim),
                slots_prev.reshape(-1, self.slot_dim)
            ).reshape(B, K, self.slot_dim)
            
            # MLP 残差
            slots = slots + self.mlp(self.norm_mlp(slots))
        
        # 最终注意力权重 (用于生成 masks)
        slots_normed = self.norm_slots(slots)
        q = self.to_q(slots_normed)
        attn_logits = torch.einsum('bkd,bnd->bkn', q, k) * self.scale
        final_attn = F.softmax(attn_logits, dim=1)  # [B, K, N]
        
        return slots, final_attn


class SlotAttentionHead(nn.Module):
    """
    完整的 Slot Attention Head
    包含 Slot Attention + Mask Decoder
    """
    
    def __init__(
        self,
        input_dim: int = 256,
        num_slots: int = 7,
        slot_dim: int = 64,
        hidden_dim: int = 128,
        num_iterations: int = 3
    ):
        """
        Args:
            input_dim: 输入特征维度 (FlowFormer context 是 256)
            num_slots: slot 数量，建议 5-10
            slot_dim: slot 特征维度
            hidden_dim: MLP 隐藏层维度
            num_iterations: Slot Attention 迭代次数
        """
        super().__init__()
        
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        
        # Slot Attention 核心模块
        self.slot_attention = SlotAttention(
            num_slots=num_slots,
            input_dim=input_dim,
            slot_dim=slot_dim,
            hidden_dim=hidden_dim,
            num_iterations=num_iterations
        )
        
        # Spatial Broadcast Decoder: 将 slot 特征广播到空间维度后解码
        self.slot_proj = nn.Linear(slot_dim, slot_dim)
        
        # 简化的 Mask Decoder
        self.mask_mlp = nn.Sequential(
            nn.Conv2d(slot_dim + 2, hidden_dim, kernel_size=3, padding=1),  # +2 for positional encoding
            nn.GroupNorm(8, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim // 2, 1, kernel_size=1)
        )
        
    def _create_pos_encoding(self, H, W, device):
        """创建位置编码"""
        y = torch.linspace(-1, 1, H, device=device)
        x = torch.linspace(-1, 1, W, device=device)
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        pos = torch.stack([xx, yy], dim=0)  # [2, H, W]
        return pos
        
    def forward(self, features, feature_size):
        """
        Args:
            features: [B, C, H, W] 输入特征图
            feature_size: (H, W) 特征图尺寸
            
        Returns:
            slots: [B, K, slot_dim] slot 特征
            masks: [B, K, H_out, W_out] 分割 masks (上采样后)
            attn_masks: [B, K, H, W] 原始分辨率的注意力 masks
        """
        B, C, H, W = features.shape
        
        # 展平特征图
        features_flat = rearrange(features, 'b c h w -> b (h w) c')
        
        # Slot Attention
        slots, attn_weights = self.slot_attention(features_flat)  # [B, K, D], [B, K, H*W]
        
        # 重塑注意力权重为空间形式
        attn_masks = rearrange(attn_weights, 'b k (h w) -> b k h w', h=H, w=W)
        
        # Spatial Broadcast Decoder
        # 将 slots 广播到空间维度
        slots_proj = self.slot_proj(slots)  # [B, K, D]
        slots_broadcast = repeat(slots_proj, 'b k d -> b k d h w', h=H, w=W)
        
        # 添加位置编码
        pos_enc = self._create_pos_encoding(H, W, features.device)  # [2, H, W]
        pos_enc = repeat(pos_enc, 'c h w -> b k c h w', b=B, k=self.num_slots)
        
        # 拼接 slot 特征和位置编码
        decoder_input = torch.cat([slots_broadcast, pos_enc], dim=2)  # [B, K, D+2, H, W]
        
        # 逐 slot 解码 masks
        masks_list = []
        for k in range(self.num_slots):
            slot_input = decoder_input[:, k]  # [B, D+2, H, W]
            mask_k = self.mask_mlp(slot_input)  # [B, 1, H, W]
            masks_list.append(mask_k)
        
        masks = torch.cat(masks_list, dim=1)  # [B, K, H, W]
        masks = F.softmax(masks, dim=1)  # 归一化，确保 masks 和为 1
        
        return slots, masks, attn_masks
