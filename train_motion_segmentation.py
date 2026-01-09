"""
Motion Segmentation 训练脚本
分阶段训练策略:
- Phase 1: 冻结 FlowFormer，只训练 Slot Attention + Motion Decoder
- Phase 2: 解冻 Encoder 后几层，联合优化
"""

from __future__ import print_function, division
import argparse
import os
import time
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

# 解决 "Too many open files" 问题
torch.multiprocessing.set_sharing_strategy('file_system')

from loguru import logger as loguru_logger

from core.FlowFormer import build_flowformer
from core.segmentation import MotionSegmentationModel
from core.segmentation.losses import SegmentationLoss, JointLoss
from core.utils.misc import process_cfg
import core.datasets as datasets

try:
    from torch.cuda.amp import GradScaler
except:
    class GradScaler:
        def __init__(self):
            pass
        def scale(self, loss):
            return loss
        def unscale_(self, optimizer):
            pass
        def step(self, optimizer):
            optimizer.step()
        def update(self):
            pass


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def create_model(cfg, flowformer_ckpt):
    """创建模型并加载预训练权重"""
    # 1. 构建 FlowFormer
    flowformer = build_flowformer(cfg)
    
    # 2. 加载预训练权重
    if flowformer_ckpt is not None:
        loguru_logger.info(f"Loading FlowFormer checkpoint from {flowformer_ckpt}")
        state_dict = torch.load(flowformer_ckpt)
        # 处理 DataParallel 的 state_dict
        if list(state_dict.keys())[0].startswith('module.'):
            state_dict = {k[7:]: v for k, v in state_dict.items()}
        flowformer.load_state_dict(state_dict, strict=True)
    
    # 3. 构建完整模型
    model = MotionSegmentationModel(
        flowformer_model=flowformer,
        num_slots=cfg.num_slots,
        slot_dim=cfg.slot_dim,
        hidden_dim=cfg.hidden_dim,
        num_iterations=cfg.slot_iterations,
        motion_model=cfg.motion_model,
        feature_scale=cfg.feature_scale
    )
    
    return model


def train_phase1(model, train_loader, cfg, logger_path):
    """
    Phase 1: 热身阶段
    - 冻结 FlowFormer 的 Encoder 和 Flow Head
    - 只训练 SlotAttention 和 MotionDecoder
    - Loss = MSE(Flow_Recon, Flow_Pred.detach())
    """
    loguru_logger.info("=" * 50)
    loguru_logger.info("Phase 1: Training Segmentation Head (Frozen Backbone)")
    loguru_logger.info("=" * 50)
    
    # 冻结 FlowFormer
    model.freeze_flowformer()
    
    # 获取可训练参数
    trainable_params = model.get_trainable_params(phase='phase1')
    loguru_logger.info(f"Trainable parameters: {sum(p.numel() for p in trainable_params)}")
    
    # 优化器
    optimizer = optim.AdamW(trainable_params, lr=cfg.phase1_lr, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=cfg.phase1_lr,
        total_steps=cfg.phase1_steps,
        pct_start=0.1
    )
    
    # 损失函数 - 初期只用重建损失，减少正则化干扰
    criterion = SegmentationLoss(
        recon_weight=1.0,
        mask_reg_weight=0.01,  # 降低正则化权重
        motion_consist_weight=0.0  # 初期关闭运动一致性损失
    )
    
    scaler = GradScaler(enabled=cfg.mixed_precision)
    
    model.cuda()
    model.train()
    
    total_steps = 0
    running_loss = 0.0
    running_recon = 0.0
    running_mask_reg = 0.0
    
    while total_steps < cfg.phase1_steps:
        for i_batch, data_blob in enumerate(train_loader):
            image1, image2, flow_gt, valid = [x.cuda() for x in data_blob]
            
            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast(enabled=cfg.mixed_precision):
                # 前向传播
                output = model(image1, image2, return_flow=True)
                
                # 使用 FlowFormer 的预测作为目标 (detach!)
                if 'flow_predictions' in output:
                    flow_target = output['flow_predictions'][-1].detach()
                else:
                    flow_target = output['flow_pred'].detach()
                
                # 调试信息 (前几个 step)
                if total_steps < 5:
                    loguru_logger.info(f"  flow_target range: [{flow_target.min():.2f}, {flow_target.max():.2f}]")
                    loguru_logger.info(f"  flow_recon range: [{output['flow_recon'].min():.2f}, {output['flow_recon'].max():.2f}]")
                    loguru_logger.info(f"  masks sum: {output['masks'].sum(dim=1).mean():.4f}")
                    # 显示缩放因子
                    if hasattr(model.motion_decoder.motion_model, 'residual_scale'):
                        loguru_logger.info(f"  residual_scale: {model.motion_decoder.motion_model.residual_scale.item():.2f}")
                    if hasattr(model.motion_decoder.motion_model, 'global_scale'):
                        loguru_logger.info(f"  global_scale: {model.motion_decoder.motion_model.global_scale.item():.2f}")
                
                # 每 1000 步打印一次详细信息
                if total_steps > 0 and total_steps % 1000 == 0:
                    with torch.no_grad():
                        flow_err = (output['flow_recon'] - flow_target).abs().mean()
                        flow_recon_mag = output['flow_recon'].abs().mean()
                        flow_target_mag = flow_target.abs().mean()
                        
                        # 监控缩放因子
                        scale_info = ""
                        if hasattr(model.motion_decoder.motion_model, 'residual_scale'):
                            res_scale = model.motion_decoder.motion_model.residual_scale.item()
                            scale_info += f"res_scale: {res_scale:.2f}, "
                        if hasattr(model.motion_decoder.motion_model, 'global_scale'):
                            glob_scale = model.motion_decoder.motion_model.global_scale.item()
                            scale_info += f"glob_scale: {glob_scale:.2f}"
                            
                    loguru_logger.info(
                        f"  [详细] flow_recon_mag: {flow_recon_mag:.2f}, "
                        f"flow_target_mag: {flow_target_mag:.2f}, "
                        f"avg_err: {flow_err:.2f}, {scale_info}"
                    )
                
                # 计算损失
                loss, loss_dict = criterion(output, flow_target)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable_params, cfg.clip)
            scaler.step(optimizer)
            scheduler.step()
            scaler.update()
            
            running_loss += loss.item()
            running_recon += loss_dict['loss_recon']
            running_mask_reg += loss_dict.get('loss_mask_reg', 0)
            total_steps += 1
            
            if total_steps % cfg.log_freq == 0:
                avg_loss = running_loss / cfg.log_freq
                avg_recon = running_recon / cfg.log_freq
                avg_mask_reg = running_mask_reg / cfg.log_freq
                lr = scheduler.get_last_lr()[0]
                loguru_logger.info(
                    f"Phase1 Step {total_steps}/{cfg.phase1_steps} | "
                    f"Loss: {avg_loss:.4f} | Recon: {avg_recon:.4f} | "
                    f"MaskReg: {avg_mask_reg:.4f} | LR: {lr:.6f}"
                )
                running_loss = 0.0
                running_recon = 0.0
                running_mask_reg = 0.0
                
            if total_steps % cfg.save_freq == 0:
                save_path = os.path.join(logger_path, f'phase1_step{total_steps}.pth')
                torch.save(model.state_dict(), save_path)
                loguru_logger.info(f"Saved checkpoint to {save_path}")
                
            if total_steps >= cfg.phase1_steps:
                break
    
    # 保存 Phase 1 最终模型
    save_path = os.path.join(logger_path, 'phase1_final.pth')
    torch.save(model.state_dict(), save_path)
    loguru_logger.info(f"Phase 1 completed. Saved to {save_path}")
    
    return model


def train_phase2(model, train_loader, cfg, logger_path):
    """
    Phase 2: 联合优化阶段
    - 解冻 Encoder 的后几层
    - 联合优化光流和分割
    - Loss = Weight_Flow * L_Flow + Weight_Seg * L_Recon
    """
    loguru_logger.info("=" * 50)
    loguru_logger.info("Phase 2: Joint Training (Partial Unfreezing)")
    loguru_logger.info("=" * 50)
    
    # 解冻 Encoder 后几层
    model.unfreeze_encoder_partial(num_layers=cfg.unfreeze_layers)
    
    # 获取可训练参数
    trainable_params = model.get_trainable_params(phase='phase2')
    loguru_logger.info(f"Trainable parameters: {sum(p.numel() for p in trainable_params)}")
    
    # 优化器 - 使用不同学习率
    encoder_params = []
    head_params = []
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'flowformer_wrapper' in name:
                encoder_params.append(param)
            else:
                head_params.append(param)
    
    optimizer = optim.AdamW([
        {'params': encoder_params, 'lr': cfg.phase2_lr * 0.1},  # Encoder 用更小的学习率
        {'params': head_params, 'lr': cfg.phase2_lr}
    ], weight_decay=cfg.weight_decay)
    
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[cfg.phase2_lr * 0.1, cfg.phase2_lr],
        total_steps=cfg.phase2_steps,
        pct_start=0.1
    )
    
    # 联合损失函数
    criterion = JointLoss(
        flow_weight=cfg.flow_weight,
        seg_weight=cfg.seg_weight,
        gamma=cfg.gamma,
        max_flow=cfg.max_flow
    )
    
    scaler = GradScaler(enabled=cfg.mixed_precision)
    
    model.cuda()
    model.train()
    
    total_steps = 0
    running_loss = 0.0
    
    while total_steps < cfg.phase2_steps:
        for i_batch, data_blob in enumerate(train_loader):
            image1, image2, flow_gt, valid = [x.cuda() for x in data_blob]
            
            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast(enabled=cfg.mixed_precision):
                output = model(image1, image2, return_flow=True)
                loss, loss_dict = criterion(output, flow_gt, valid)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable_params, cfg.clip)
            scaler.step(optimizer)
            scheduler.step()
            scaler.update()
            
            running_loss += loss.item()
            total_steps += 1
            
            if total_steps % cfg.log_freq == 0:
                avg_loss = running_loss / cfg.log_freq
                loguru_logger.info(
                    f"Phase2 Step {total_steps}/{cfg.phase2_steps} | "
                    f"Loss: {avg_loss:.4f} | "
                    f"Flow: {loss_dict.get('loss_flow', 0):.4f} | "
                    f"Recon: {loss_dict['loss_recon']:.4f}"
                )
                running_loss = 0.0
                
            if total_steps % cfg.save_freq == 0:
                save_path = os.path.join(logger_path, f'phase2_step{total_steps}.pth')
                torch.save(model.state_dict(), save_path)
                
            if total_steps >= cfg.phase2_steps:
                break
    
    # 保存最终模型
    save_path = os.path.join(logger_path, 'final_model.pth')
    torch.save(model.state_dict(), save_path)
    loguru_logger.info(f"Phase 2 completed. Saved to {save_path}")
    
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='motion_seg', help="experiment name")
    parser.add_argument('--stage', default='things', help="dataset stage")
    parser.add_argument('--flowformer_ckpt', type=str, required=True, help="path to FlowFormer checkpoint")
    
    # 模型参数
    parser.add_argument('--num_slots', type=int, default=7, help="number of slots")
    parser.add_argument('--slot_dim', type=int, default=64, help="slot dimension")
    parser.add_argument('--hidden_dim', type=int, default=128, help="hidden dimension")
    parser.add_argument('--slot_iterations', type=int, default=3, help="slot attention iterations")
    parser.add_argument('--motion_model', type=str, default='hybrid', 
                        choices=['dense', 'cnn', 'hybrid'],
                        help="motion model type: dense (MLP), cnn (CNN decoder), hybrid (homography + residual)")
    parser.add_argument('--feature_scale', type=str, default='1/8', choices=['1/4', '1/8'])
    
    # Phase 1 参数
    parser.add_argument('--phase1_steps', type=int, default=50000, help="phase 1 training steps")
    parser.add_argument('--phase1_lr', type=float, default=1e-3, help="phase 1 learning rate")
    
    # Phase 2 参数
    parser.add_argument('--phase2_steps', type=int, default=100000, help="phase 2 training steps")
    parser.add_argument('--phase2_lr', type=float, default=1e-4, help="phase 2 learning rate")
    parser.add_argument('--unfreeze_layers', type=int, default=2, help="number of encoder layers to unfreeze")
    parser.add_argument('--flow_weight', type=float, default=1.0, help="flow loss weight")
    parser.add_argument('--seg_weight', type=float, default=0.5, help="segmentation loss weight")
    
    # 通用参数
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--clip', type=float, default=1.0)
    parser.add_argument('--gamma', type=float, default=0.8)
    parser.add_argument('--max_flow', type=float, default=400)
    parser.add_argument('--mixed_precision', action='store_true')
    parser.add_argument('--log_freq', type=int, default=100)
    parser.add_argument('--save_freq', type=int, default=5000)
    parser.add_argument('--skip_phase1', action='store_true', help="skip phase 1")
    parser.add_argument('--phase1_ckpt', type=str, default=None, help="phase 1 checkpoint to resume")
    
    args = parser.parse_args()
    
    # 加载基础配置
    if args.stage == 'chairs':
        from configs.default import get_cfg
    elif args.stage == 'things':
        from configs.things import get_cfg
    elif args.stage == 'sintel':
        from configs.sintel import get_cfg
    elif args.stage == 'kitti':
        from configs.kitti import get_cfg
    else:
        from configs.default import get_cfg
    
    cfg = get_cfg()
    cfg.update(vars(args))
    process_cfg(cfg)
    
    # 创建日志目录
    log_dir = Path(f'checkpoints/motion_seg/{args.name}')
    log_dir.mkdir(parents=True, exist_ok=True)
    
    loguru_logger.add(str(log_dir / 'log.txt'), encoding="utf8")
    loguru_logger.info(cfg)
    
    # 设置随机种子
    torch.manual_seed(1234)
    np.random.seed(1234)
    
    # 创建模型
    model = create_model(cfg, args.flowformer_ckpt)
    loguru_logger.info(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    
    # 加载数据 - 使用较少的 workers 避免 "Too many open files" 错误
    import torch.utils.data as data
    
    # 临时修改 num_workers
    original_fetch = datasets.fetch_dataloader
    def fetch_dataloader_safe(args, TRAIN_DS='C+T+K+S+H'):
        loader = original_fetch(args, TRAIN_DS)
        # 重新创建 DataLoader 使用较少的 workers
        return data.DataLoader(
            loader.dataset, 
            batch_size=args.batch_size,
            pin_memory=False, 
            shuffle=True, 
            num_workers=16,  # 减少 worker 数量
            drop_last=True
        )
    
    train_loader = fetch_dataloader_safe(cfg)
    
    # Phase 1
    if not args.skip_phase1:
        if args.phase1_ckpt is not None:
            loguru_logger.info(f"Loading Phase 1 checkpoint from {args.phase1_ckpt}")
            model.load_state_dict(torch.load(args.phase1_ckpt))
        model = train_phase1(model, train_loader, cfg, str(log_dir))
    else:
        if args.phase1_ckpt is not None:
            loguru_logger.info(f"Loading Phase 1 checkpoint from {args.phase1_ckpt}")
            model.load_state_dict(torch.load(args.phase1_ckpt))
    
    # Phase 2
    model = train_phase2(model, train_loader, cfg, str(log_dir))
    
    loguru_logger.info("Training completed!")


if __name__ == '__main__':
    main()
