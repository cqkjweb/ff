# Motion Segmentation Module for FlowFormer
# Self-supervised motion segmentation using Slot Attention
# Supports complex non-aligned viewpoint changes

from .wrapper import FlowFormerWrapper
from .slot_attention import SlotAttentionHead
from .motion_decoder import MotionDecoder, DenseFlowDecoder, CNNFlowDecoder, HybridMotionDecoder
from .segmentation_model import MotionSegmentationModel

__all__ = [
    'FlowFormerWrapper',
    'SlotAttentionHead', 
    'MotionDecoder',
    'DenseFlowDecoder',
    'CNNFlowDecoder', 
    'HybridMotionDecoder',
    'MotionSegmentationModel'
]
