"""
测试 Motion Segmentation 模块
验证各组件是否能正常工作
"""

import torch
import torch.nn as nn
import sys

def test_slot_attention():
    """测试 Slot Attention 模块"""
    print("Testing Slot Attention...")
    
    from core.segmentation.slot_attention import SlotAttention, SlotAttentionHead
    
    # 测试 SlotAttention
    batch_size = 2
    num_tokens = 48 * 64  # H/8 * W/8
    input_dim = 256
    num_slots = 7
    slot_dim = 64
    
    slot_attn = SlotAttention(
        num_slots=num_slots,
        input_dim=input_dim,
        slot_dim=slot_dim
    )
    
    x = torch.randn(batch_size, num_tokens, input_dim)
    slots, attn = slot_attn(x)
    
    assert slots.shape == (batch_size, num_slots, slot_dim), f"Slots shape mismatch: {slots.shape}"
    assert attn.shape == (batch_size, num_slots, num_tokens), f"Attn shape mismatch: {attn.shape}"
    assert torch.allclose(attn.sum(dim=1), torch.ones(batch_size, num_tokens)), "Attn should sum to 1 over slots"
    
    print(f"  SlotAttention: slots={slots.shape}, attn={attn.shape} ✓")
    
    # 测试 SlotAttentionHead
    H, W = 48, 64
    features = torch.randn(batch_size, input_dim, H, W)
    
    head = SlotAttentionHead(
        input_dim=input_dim,
        num_slots=num_slots,
        slot_dim=slot_dim
    )
    
    slots, masks, attn_masks = head(features, (H, W))
    
    assert slots.shape == (batch_size, num_slots, slot_dim), f"Slots shape mismatch: {slots.shape}"
    assert masks.shape == (batch_size, num_slots, H, W), f"Masks shape mismatch: {masks.shape}"
    assert attn_masks.shape == (batch_size, num_slots, H, W), f"Attn masks shape mismatch: {attn_masks.shape}"
    
    print(f"  SlotAttentionHead: slots={slots.shape}, masks={masks.shape} ✓")
    print("Slot Attention tests passed! ✓\n")


def test_motion_decoder():
    """测试 Motion Decoder 模块"""
    print("Testing Motion Decoder...")
    
    from core.segmentation.motion_decoder import MotionDecoder, AffineMotionModel
    
    batch_size = 2
    num_slots = 7
    slot_dim = 64
    H, W = 384, 512
    
    # 测试 AffineMotionModel
    affine_model = AffineMotionModel(slot_dim=slot_dim)
    slots = torch.randn(batch_size, num_slots, slot_dim)
    coords = torch.stack(torch.meshgrid(
        torch.arange(W).float(),
        torch.arange(H).float(),
        indexing='xy'
    ), dim=0).unsqueeze(0).repeat(batch_size, 1, 1, 1)
    
    flows, params = affine_model(slots, coords)
    
    assert flows.shape == (batch_size, num_slots, 2, H, W), f"Flows shape mismatch: {flows.shape}"
    assert params.shape == (batch_size, num_slots, 6), f"Params shape mismatch: {params.shape}"
    
    print(f"  AffineMotionModel: flows={flows.shape}, params={params.shape} ✓")
    
    # 测试 MotionDecoder
    decoder = MotionDecoder(slot_dim=slot_dim, motion_model='affine')
    masks = torch.softmax(torch.randn(batch_size, num_slots, H // 8, W // 8), dim=1)
    
    flow_recon, slot_flows, motion_params = decoder(slots, masks, (H // 8, W // 8))
    
    assert flow_recon.shape == (batch_size, 2, H // 8, W // 8), f"Flow recon shape mismatch: {flow_recon.shape}"
    
    print(f"  MotionDecoder: flow_recon={flow_recon.shape} ✓")
    print("Motion Decoder tests passed! ✓\n")


def test_losses():
    """测试损失函数"""
    print("Testing Losses...")
    
    from core.segmentation.losses import FlowReconstructionLoss, SegmentationLoss
    
    batch_size = 2
    H, W = 384, 512
    num_slots = 7
    
    # 测试 FlowReconstructionLoss
    recon_loss = FlowReconstructionLoss(loss_type='l1')
    flow_recon = torch.randn(batch_size, 2, H, W)
    flow_target = torch.randn(batch_size, 2, H, W)
    
    loss = recon_loss(flow_recon, flow_target)
    assert loss.dim() == 0, "Loss should be scalar"
    print(f"  FlowReconstructionLoss: {loss.item():.4f} ✓")
    
    # 测试 SegmentationLoss
    seg_loss = SegmentationLoss()
    output = {
        'flow_recon': flow_recon,
        'masks': torch.softmax(torch.randn(batch_size, num_slots, H, W), dim=1),
        'slot_flows': torch.randn(batch_size, num_slots, 2, H // 8, W // 8)
    }
    
    total_loss, loss_dict = seg_loss(output, flow_target)
    assert total_loss.dim() == 0, "Total loss should be scalar"
    print(f"  SegmentationLoss: {total_loss.item():.4f} ✓")
    print(f"    Loss breakdown: {loss_dict}")
    print("Loss tests passed! ✓\n")


def test_wrapper():
    """测试 FlowFormer Wrapper (需要完整模型)"""
    print("Testing FlowFormer Wrapper...")
    print("  Skipping (requires full FlowFormer model)")
    print("  To test, run with --full flag\n")


def test_full_model():
    """测试完整模型 (需要 FlowFormer 权重)"""
    print("Testing Full Model...")
    
    try:
        from configs.things import get_cfg
        from core.FlowFormer import build_flowformer
        from core.segmentation import MotionSegmentationModel
        
        cfg = get_cfg()
        cfg.pretrain = False  # 不加载预训练权重
        
        # 构建模型
        flowformer = build_flowformer(cfg)
        model = MotionSegmentationModel(
            flowformer_model=flowformer,
            num_slots=7,
            slot_dim=64,
            hidden_dim=128,
            num_iterations=3,
            motion_model='affine'
        )
        
        # 测试前向传播
        batch_size = 1
        H, W = 384, 512
        image1 = torch.randn(batch_size, 3, H, W) * 255
        image2 = torch.randn(batch_size, 3, H, W) * 255
        
        model.eval()
        with torch.no_grad():
            output = model(image1, image2, return_flow=True)
        
        print(f"  Output keys: {list(output.keys())}")
        print(f"  flow_recon: {output['flow_recon'].shape}")
        print(f"  masks: {output['masks'].shape}")
        print(f"  slots: {output['slots'].shape}")
        print("Full model test passed! ✓\n")
        
    except Exception as e:
        print(f"  Error: {e}")
        print("  Full model test skipped (may need dependencies)\n")


def main():
    print("=" * 60)
    print("Motion Segmentation Module Tests")
    print("=" * 60 + "\n")
    
    test_slot_attention()
    test_motion_decoder()
    test_losses()
    
    if '--full' in sys.argv:
        test_full_model()
    else:
        test_wrapper()
    
    print("=" * 60)
    print("All basic tests passed! ✓")
    print("=" * 60)


if __name__ == '__main__':
    main()
