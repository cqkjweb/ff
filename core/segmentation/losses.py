"""
损失函数设计 - 自监督运动分割
核心思想: 物体 = 具有一致运动模式的区域
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FlowReconstructionLoss(nn.Module):
    """
    光流重建损失
    如果 Slot Attention 生成的 Mask 是正确的，
    那么利用这些 Mask 和每个 Slot 代表的运动参数，
    应该能完美重建出 FlowFormer 预测的光流场
    """
    
    def __init__(self, loss_type='l1'):
        super().__init__()
        self.loss_type = loss_type
        
    def forward(self, flow_recon, flow_target, valid_mask=None):
        """
        Args:
            flow_recon: [B, 2, H, W] 重建的光流
            flow_target: [B, 2, H, W] 目标光流 (FlowFormer 预测或 GT)
            valid_mask: [B, 1, H, W] 可选的有效区域 mask
            
        Returns:
            loss: 标量损失值
        """
        if self.loss_type == 'l1':
            diff = (flow_recon - flow_target).abs()
        elif self.loss_type == 'l2':
            diff = (flow_recon - flow_target) ** 2
        elif self.loss_type == 'charbonnier':
            diff = (flow_recon - flow_target) ** 2
            diff = torch.sqrt(diff + 1e-6)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
            
        if valid_mask is not None:
            diff = diff * valid_mask
            loss = diff.sum() / (valid_mask.sum() * 2 + 1e-8)
        else:
            loss = diff.mean()
            
        return loss


class MaskRegularizationLoss(nn.Module):
    """
    Mask 正则化损失
    鼓励 masks 具有良好的性质
    """
    
    def __init__(self):
        super().__init__()
        
    def entropy_loss(self, masks):
        """
        熵损失: 鼓励 masks 更加确定 (接近 0 或 1)
        """
        eps = 1e-8
        entropy = -masks * torch.log(masks + eps)
        return entropy.mean()
    
    def area_balance_loss(self, masks, min_area=0.01):
        """
        面积平衡损失: 防止某些 slots 退化为空
        """
        B, K, H, W = masks.shape
        areas = masks.sum(dim=(-2, -1)) / (H * W)  # [B, K]
        
        # 惩罚面积过小的 slots
        small_area_penalty = F.relu(min_area - areas).mean()
        
        return small_area_penalty
    
    def smoothness_loss(self, masks):
        """
        平滑损失: 鼓励空间连续的 masks
        """
        # 水平梯度
        grad_x = (masks[:, :, :, 1:] - masks[:, :, :, :-1]).abs()
        # 垂直梯度
        grad_y = (masks[:, :, 1:, :] - masks[:, :, :-1, :]).abs()
        
        return grad_x.mean() + grad_y.mean()
    
    def forward(self, masks, weights=None):
        """
        Args:
            masks: [B, K, H, W] 分割 masks
            weights: dict of loss weights
        """
        if weights is None:
            weights = {'entropy': 0.1, 'area': 0.1, 'smooth': 0.01}
            
        loss = 0
        
        if weights.get('entropy', 0) > 0:
            loss += weights['entropy'] * self.entropy_loss(masks)
            
        if weights.get('area', 0) > 0:
            loss += weights['area'] * self.area_balance_loss(masks)
            
        if weights.get('smooth', 0) > 0:
            loss += weights['smooth'] * self.smoothness_loss(masks)
            
        return loss


class MotionConsistencyLoss(nn.Module):
    """
    运动一致性损失
    同一个 slot 内的像素应该具有相似的运动
    """
    
    def __init__(self):
        super().__init__()
        
    def forward(self, slot_flows, masks, flow_target):
        """
        Args:
            slot_flows: [B, K, 2, H, W] 每个 slot 的运动场
            masks: [B, K, H, W] 分割 masks
            flow_target: [B, 2, H, W] 目标光流
            
        Returns:
            loss: 运动一致性损失
        """
        B, K, _, H, W = slot_flows.shape
        
        # 计算每个 slot 内的运动方差
        total_loss = 0
        
        for k in range(K):
            mask_k = masks[:, k:k+1]  # [B, 1, H, W]
            flow_k = slot_flows[:, k]  # [B, 2, H, W]
            
            # 加权均值
            weighted_flow = flow_k * mask_k
            mean_flow = weighted_flow.sum(dim=(-2, -1), keepdim=True) / (mask_k.sum(dim=(-2, -1), keepdim=True) + 1e-8)
            
            # 加权方差
            variance = ((flow_k - mean_flow) ** 2 * mask_k).sum(dim=(-2, -1)) / (mask_k.sum(dim=(-2, -1)) + 1e-8)
            total_loss += variance.mean()
            
        return total_loss / K


class SegmentationLoss(nn.Module):
    """
    完整的分割损失函数
    结合多种损失项
    """
    
    def __init__(
        self,
        recon_weight: float = 1.0,
        mask_reg_weight: float = 0.1,
        motion_consist_weight: float = 0.1,
        loss_type: str = 'l1'
    ):
        super().__init__()
        
        self.recon_weight = recon_weight
        self.mask_reg_weight = mask_reg_weight
        self.motion_consist_weight = motion_consist_weight
        
        self.recon_loss = FlowReconstructionLoss(loss_type)
        self.mask_reg_loss = MaskRegularizationLoss()
        self.motion_consist_loss = MotionConsistencyLoss()
        
    def forward(self, output, flow_target, valid_mask=None):
        """
        Args:
            output: 模型输出 dict
            flow_target: [B, 2, H, W] 目标光流
            valid_mask: [B, 1, H, W] 可选的有效区域
            
        Returns:
            total_loss: 总损失
            loss_dict: 各项损失的字典
        """
        flow_recon = output['flow_recon']
        masks = output['masks']
        slot_flows = output.get('slot_flows')
        
        # 确保尺寸匹配
        if flow_recon.shape[-2:] != flow_target.shape[-2:]:
            flow_recon = F.interpolate(
                flow_recon, size=flow_target.shape[-2:],
                mode='bilinear', align_corners=False
            )
            
        if masks.shape[-2:] != flow_target.shape[-2:]:
            masks = F.interpolate(
                masks, size=flow_target.shape[-2:],
                mode='bilinear', align_corners=False
            )
            masks = F.softmax(masks, dim=1)
            
        # 1. 重建损失
        loss_recon = self.recon_loss(flow_recon, flow_target, valid_mask)
        
        # 2. Mask 正则化
        loss_mask_reg = self.mask_reg_loss(masks)
        
        # 3. 运动一致性 (可选)
        loss_motion = 0
        if slot_flows is not None and self.motion_consist_weight > 0:
            # 调整 slot_flows 尺寸
            B, K, _, H_s, W_s = slot_flows.shape
            H, W = flow_target.shape[-2:]
            slot_flows_resized = F.interpolate(
                slot_flows.view(B*K, 2, H_s, W_s),
                size=(H, W), mode='bilinear', align_corners=False
            ).view(B, K, 2, H, W)
            loss_motion = self.motion_consist_loss(slot_flows_resized, masks, flow_target)
        
        # 总损失
        total_loss = (
            self.recon_weight * loss_recon +
            self.mask_reg_weight * loss_mask_reg +
            self.motion_consist_weight * loss_motion
        )
        
        loss_dict = {
            'loss_total': total_loss.item(),
            'loss_recon': loss_recon.item(),
            'loss_mask_reg': loss_mask_reg.item() if isinstance(loss_mask_reg, torch.Tensor) else loss_mask_reg,
            'loss_motion': loss_motion.item() if isinstance(loss_motion, torch.Tensor) else loss_motion
        }
        
        return total_loss, loss_dict


class JointLoss(nn.Module):
    """
    联合训练损失 (Phase 2)
    结合光流损失和分割损失
    """
    
    def __init__(
        self,
        flow_weight: float = 1.0,
        seg_weight: float = 0.5,
        gamma: float = 0.8,
        max_flow: float = 400
    ):
        super().__init__()
        
        self.flow_weight = flow_weight
        self.seg_weight = seg_weight
        self.gamma = gamma
        self.max_flow = max_flow
        
        self.seg_loss = SegmentationLoss()
        
    def flow_sequence_loss(self, flow_preds, flow_gt, valid):
        """FlowFormer 的序列损失"""
        n_predictions = len(flow_preds)
        flow_loss = 0.0
        
        mag = torch.sum(flow_gt**2, dim=1).sqrt()
        valid = (valid >= 0.5) & (mag < self.max_flow)
        
        for i in range(n_predictions):
            i_weight = self.gamma ** (n_predictions - i - 1)
            i_loss = (flow_preds[i] - flow_gt).abs()
            flow_loss += i_weight * (valid[:, None] * i_loss).mean()
            
        return flow_loss
        
    def forward(self, output, flow_gt, valid):
        """
        Args:
            output: 模型输出
            flow_gt: [B, 2, H, W] 光流 GT
            valid: [B, H, W] 有效 mask
            
        Returns:
            total_loss: 总损失
            loss_dict: 各项损失
        """
        loss_dict = {}
        
        # 1. 光流损失 (如果有 GT)
        loss_flow = 0
        if 'flow_predictions' in output:
            loss_flow = self.flow_sequence_loss(
                output['flow_predictions'], flow_gt, valid
            )
            loss_dict['loss_flow'] = loss_flow.item()
            
        # 2. 分割损失 (使用 FlowFormer 预测作为目标)
        # 在 Phase 2，我们用 GT 光流作为目标
        flow_target = flow_gt
        valid_mask = (valid >= 0.5).unsqueeze(1).float()
        
        loss_seg, seg_loss_dict = self.seg_loss(output, flow_target, valid_mask)
        loss_dict.update(seg_loss_dict)
        
        # 总损失
        total_loss = self.flow_weight * loss_flow + self.seg_weight * loss_seg
        loss_dict['loss_total'] = total_loss.item()
        
        return total_loss, loss_dict
