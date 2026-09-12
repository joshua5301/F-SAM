"""Small ViTs for CIFAR trained from scratch (timm VisionTransformer, 32x32 input, patch 4)."""
from functools import partial

import torch.nn as nn
from timm.models.vision_transformer import VisionTransformer

__all__ = ['vit_tiny', 'vit_small']

_common = dict(img_size=32, patch_size=4, mlp_ratio=4, qkv_bias=True,
               norm_layer=partial(nn.LayerNorm, eps=1e-6), drop_path_rate=0.1)


class vit_tiny:                      # ~2.7M params
    base = VisionTransformer
    args = list()
    kwargs = dict(embed_dim=192, depth=9, num_heads=3, **_common)


class vit_small:                     # ~11M params
    base = VisionTransformer
    args = list()
    kwargs = dict(embed_dim=384, depth=8, num_heads=6, **_common)
