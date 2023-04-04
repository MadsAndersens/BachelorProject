import torch
import torch.nn as nn
import torch.nn.functional as F
from vit_pytorch import ViT

v = ViT(
    image_size = 320,
    patch_size = 32,
    num_classes = 2,
    dim = 1024,
    depth = 6,
    heads = 16,
    mlp_dim = 2048,
    dropout = 0.1,
    emb_dropout = 0.1
)

