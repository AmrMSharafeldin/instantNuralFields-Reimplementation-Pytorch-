import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.nn import functional as F
import einops
from einops import rearrange, reduce, repeat
from tqdm import tqdm
from nerf.utils import trilinear_interpolation


class NGLOD(torch.nn.Module):

    def __init__(self, base_lod , num_lod , feature_dim , L = 4 , scene_scale = 3):
        super(NGLOD, self).__init__()
        self.device = self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("hello")

        self.feature_dim = feature_dim 
        self.L = L  # For encoding directions
        self.scene_scale = scene_scale
        self.codebook = nn.ParameterList([])
        self.LODS = [2**L for L in range(base_lod, base_lod + num_lod)]
        print(self.LODS)
        self.init_feature_structure()
        self.sigma_mlp = nn.Sequential(nn.Linear(self.feature_dim * len(self.LODS), 64),
                                         nn.ReLU(), nn.Linear(64, 16)).to(self.device)

        self.pred_color_mlp = nn.Sequential(nn.Linear((L-1)**3 + 16, 64), nn.ReLU(),
                                       nn.Linear(64, 64), nn.ReLU(),
                                       nn.Linear(64, 3), nn.Sigmoid()).to(self.device)
    def positional_encoding(self, x):
        out = [x]
        for j in range(self.L):
            out.append(torch.sin(2 ** j * x))
            out.append(torch.cos(2 ** j * x))
        return torch.cat(out, dim=1)



    def init_feature_structure(self):
      for LOD in self.LODS:
          fts = torch.zeros(LOD**3, self.feature_dim)
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
        all_features = all_features.reshape(-1, self.feature_dim * len(self.LODS))
        embdeded_dics = self.positional_encoding(d[mask])
        h = self.sigma_mlp(all_features)
        color = torch.zeros((pts.shape[0], 3), device=pts.device)
        log_sigma = torch.zeros((pts.shape[0]), device=pts.device) - 100000
        log_sigma[mask] = h[:, 0]
        color[mask] = self.pred_color_mlp(torch.cat((h, embdeded_dics), dim=1))
        return color, torch.exp(log_sigma).unsqueeze(-1)
    def intersect(self, ray_origins, ray_directions):
      return self.forward(ray_origins, ray_directions)
