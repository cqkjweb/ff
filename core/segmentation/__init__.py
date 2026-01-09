# Motion Segmentation Module for FlowFormer
# Self-supervised motion segmentation using Slot Attention

from .wrapper import FlowFormerWrapper
from .slot_attention import SlotAttentionHead
from .motion_decoder import MotionDecoder
from .segmentation_model import MotionSegmentationModel

__all__ = [
    'FlowFormerWrapper',
    'SlotAttentionHead', 
    'MotionDecoder',
    'MotionSegmentationModel'
]
