import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.nn import functional as F
import einops
from einops import rearrange, reduce, repeat
from tqdm import tqdm
from nerf.utils import trilinear_interpolation


class Plenoxels(torch.nn.Module):

    def __init__(self, LODS, L, scene_scale, desc=2):
        super(Plenoxels, self).__init__()
        self.desc = desc
        self.L = L  # For PE encoding
        self.scene_scale = scene_scale
        self.codebook = nn.ParameterList([])
        self.LODS = LODS
        print(self.LODS)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.init_feature_structure()

    def init_feature_structure(self):
      for LOD in self.LODS:
          fts = torch.zeros(LOD**3, self.desc)
          fts += torch.randn_like(fts) * 0.01
          fts = nn.Parameter(fts)
          self.codebook.append(fts)
          self.codebook = self.codebook.to(device = self.device)

    def forward(self, pts, d):

        pts /= self.scene_scale
        mask = (pts[:, 0].abs() < .5) & (pts[:, 1].abs() < .5) & (pts[:, 2].abs() < .5)
        pts += 0.5  # x in [0, 1]^3
        feats = []
        for i, res in enumerate(self.LODS):
          features = trilinear_interpolation(res, self.codebook[i], pts[mask], "NGLOD")
          feats.append((torch.unsqueeze(features, dim=-1)))
        all_features = torch.cat(feats, -1)
        all_features = all_features.sum(-1)

        color = torch.zeros((pts.shape[0], 3), device=pts.device)
        log_sigma = torch.zeros((pts.shape[0]), device=pts.device) - 100000
        color[mask] = features[:, 0:3]
        log_sigma[mask] = features[:, -1]
        return color, torch.exp(log_sigma).unsqueeze(-1)
    def intersect(self, ray_origins, ray_directions):
      return self.forward(ray_origins, ray_directions)