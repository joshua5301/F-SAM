"""ViTs for CIFAR trained from scratch (32x32 input, patch 4).

vit_tiny / vit_small match ViT-T / ViT-S of Mueller et al. (SAM-ON, ICML 2023):
timm VisionTransformer with all defaults (mlp_ratio 4, qkv_bias, no drop-path).
"""
from timm.models.vision_transformer import VisionTransformer

__all__ = ['vit_tiny', 'vit_small']


class vit_tiny:                      # ~5.4M params
    base = VisionTransformer
    args = list()
    kwargs = dict(img_size=32, patch_size=4, embed_dim=192, depth=12, num_heads=3)


class vit_small:                     # ~21M params
    base = VisionTransformer
    args = list()
    kwargs = dict(img_size=32, patch_size=4, embed_dim=384, depth=12, num_heads=6)
