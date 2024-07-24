import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.nn import functional as F
import einops
from einops import rearrange, reduce, repeat
from tqdm import tqdm
from nerf.utils import trilinear_interpolation



class TinyMLP(torch.nn.Module):
    def __init__(self, LODS, L, scene_scale, desc=2):
      pass

    def forward(self, pts, d):
      pass
    def intersect(self, ray_origins, ray_directions):
        raise NotImplementedError("The intersect method is not implemented for this model.")
