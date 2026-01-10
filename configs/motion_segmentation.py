"""
Motion Segmentation 配置文件
"""

from yacs.config import CfgNode as CN

_C = CN()

# FlowFormer 基础配置 (继承自 things.py)
_C.transformer = 'latentcostformer'
_C.pretrain = True
_C.context_concat = False
_C.gamma = 0.8
_C.max_flow = 400

# LatentCostFormer 配置
_C.latentcostformer = CN()
_C.latentcostformer.pe = 'linear'
_C.latentcostformer.encoder_latent_dim = 256
_C.latentcostformer.cost_latent_input_dim = 64
_C.latentcostformer.cost_latent_token_num = 8
_C.latentcostformer.cost_latent_dim = 128
_C.latentcostformer.dropout = 0.0
_C.latentcostformer.encoder_depth = 3
_C.latentcostformer.decoder_depth = 32
_C.latentcostformer.cost_heads_num = 1
_C.latentcostformer.patch_size = 8
_C.latentcostformer.query_latent_dim = 64
_C.latentcostformer.add_flow_token = True
_C.latentcostformer.use_mlp = False
_C.latentcostformer.vertical_conv = False
_C.latentcostformer.cost_encoder_res = True
_C.latentcostformer.vert_c_dim = 64
_C.latentcostformer.feat_cross_attn = False
_C.latentcostformer.only_global = False
_C.latentcostformer.cnet = 'twins'
_C.latentcostformer.fnet = 'twins'
_C.latentcostformer.gma = 'GMA'
_C.latentcostformer.rpe = 'element-wise'

# Motion Segmentation 配置
_C.num_slots = 7
_C.slot_dim = 64
_C.hidden_dim = 128
_C.slot_iterations = 3
_C.motion_model = 'affine'  # 'affine' or 'quadratic'
_C.feature_scale = '1/8'

# Phase 1 训练配置
_C.phase1_steps = 50000
_C.phase1_lr = 2e-4

# Phase 2 训练配置
_C.phase2_steps = 100000
_C.phase2_lr = 1e-4
_C.phase2_batch_size = 2  # Phase 2 使用更小的 batch size
_C.gradient_accumulation = 2  # 梯度累积步数
_C.unfreeze_layers = 2
_C.flow_weight = 1.0
_C.seg_weight = 0.5

# 通用训练配置
_C.batch_size = 4
_C.weight_decay = 1e-4
_C.clip = 1.0
_C.mixed_precision = True
_C.log_freq = 100
_C.save_freq = 5000

# 数据配置
_C.stage = 'things'
_C.image_size = [384, 512]
_C.add_noise = False

# Trainer 配置 (兼容原有代码)
_C.trainer = CN()
_C.trainer.scheduler = 'OneCycleLR'
_C.trainer.optimizer = 'adamw'
_C.trainer.canonical_lr = 1e-4
_C.trainer.adamw_decay = 1e-4
_C.trainer.clip = 1.0
_C.trainer.num_steps = 150000
_C.trainer.epsilon = 1e-8
_C.trainer.anneal_strategy = 'linear'


def get_cfg():
    return _C.clone()
