"""
FlowFormer Wrapper - 复用 FlowFormer 的 Encoder 特征
提取 1/4 或 1/8 分辨率的特征图用于下游分割任务
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class FlowFormerWrapper(nn.Module):
    """
    Wrapper 类，用于复用 FlowFormer 的 Encoder 特征
    
    FlowFormer 结构:
    - context_encoder (twins_svt_large): 提取上下文特征，输出 1/8 分辨率
    - feat_encoder (twins_svt_large): 提取图像特征用于 cost volume
    - memory_encoder: 编码 cost volume
    - memory_decoder: 解码光流
    
    我们提取:
    - context features: 1/8 分辨率，包含丰富的语义信息
    - cost memory: 编码后的 cost volume 特征
    """
    
    def __init__(self, flowformer_model, feature_scale='1/8'):
        """
        Args:
            flowformer_model: 预训练的 FlowFormer 模型
            feature_scale: 特征图分辨率，'1/4' 或 '1/8'
        """
        super().__init__()
        self.flowformer = flowformer_model
        self.feature_scale = feature_scale
        
        # 获取配置
        self.cfg = flowformer_model.cfg
        
        # 特征维度 (twins_svt_large 输出 256 维)
        self.context_dim = 256
        self.cost_latent_dim = self.cfg.cost_latent_dim if hasattr(self.cfg, 'cost_latent_dim') else 256
        
        # 如果需要 1/4 分辨率，添加上采样层
        if feature_scale == '1/4':
            self.upsample_context = nn.Sequential(
                nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1),
                nn.GroupNorm(32, 256),
                nn.GELU()
            )
        else:
            self.upsample_context = None

    def extract_features(self, image1, image2):
        """
        提取 FlowFormer 的中间特征
        
        Args:
            image1: [B, 3, H, W] 第一帧图像 (0-255)
            image2: [B, 3, H, W] 第二帧图像 (0-255)
            
        Returns:
            dict containing:
                - context_features: [B, C, H/s, W/s] 上下文特征
                - cost_memory: [B*H1*W1, K, D] cost volume 编码
                - flow_predictions: list of flow predictions
                - feature_size: (H1, W1) 特征图尺寸
        """
        # 图像预处理 (与 FlowFormer 一致)
        image1_norm = 2 * (image1 / 255.0) - 1.0
        image2_norm = 2 * (image2 / 255.0) - 1.0
        
        data = {}
        
        # 1. 提取上下文特征
        if self.cfg.context_concat:
            context = self.flowformer.context_encoder(
                torch.cat([image1_norm, image2_norm], dim=1)
            )
        else:
            context = self.flowformer.context_encoder(image1_norm)
        
        B, C, H1, W1 = context.shape
        
        # 2. 编码 cost memory
        cost_memory = self.flowformer.memory_encoder(
            image1_norm, image2_norm, data, context
        )
        
        # 3. 获取光流预测
        flow_predictions = self.flowformer.memory_decoder(
            cost_memory, context, data, flow_init=None
        )
        
        # 4. 可选：上采样到 1/4 分辨率
        if self.upsample_context is not None:
            context_out = self.upsample_context(context)
        else:
            context_out = context
            
        return {
            'context_features': context_out,
            'cost_memory': cost_memory,
            'flow_predictions': flow_predictions,
            'feature_size': (H1, W1),
            'data': data
        }
    
    def forward(self, image1, image2):
        """
        完整的前向传播，返回光流和特征
        """
        return self.extract_features(image1, image2)
    
    def freeze_encoder(self):
        """冻结 Encoder 参数"""
        for param in self.flowformer.context_encoder.parameters():
            param.requires_grad = False
        for param in self.flowformer.memory_encoder.parameters():
            param.requires_grad = False
            
    def unfreeze_encoder_layers(self, num_layers=2):
        """
        解冻 Encoder 的后几层
        用于 Phase 2 的微调
        """
        # 解冻 context_encoder 的后几层
        if hasattr(self.flowformer.context_encoder, 'svt'):
            blocks = self.flowformer.context_encoder.svt.blocks
            for i, block in enumerate(blocks):
                if i >= len(blocks) - num_layers:
                    for param in block.parameters():
                        param.requires_grad = True
                        
        # 解冻 memory_encoder 的 cost_perceiver_encoder 后几层
        encoder_layers = self.flowformer.memory_encoder.cost_perceiver_encoder.encoder_layers
        for i, layer in enumerate(encoder_layers):
            if i >= len(encoder_layers) - num_layers:
                for param in layer.parameters():
                    param.requires_grad = True
                    
    def freeze_flow_head(self):
        """冻结 Flow Head (memory_decoder)"""
        for param in self.flowformer.memory_decoder.parameters():
            param.requires_grad = False
