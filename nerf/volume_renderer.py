import numpy as np
import matplotlib.pyplot as plt
import torch
import einops
from einops import rearrange, reduce, repeat

def compute_accumulated_transmittance(alphas):
    accumulated_transmittance = torch.cumprod(alphas, 1)
    return torch.cat((torch.ones((accumulated_transmittance.shape[0], 1), device=alphas.device),
                      accumulated_transmittance[:, :-1]), dim=-1)

def volume_renderer (model , rays_o, rays_d, near , far , bins = 100 ):
  """
  Args:
      rays_o: (H*W, 3)
      rays_d: (H*W, 3)
      model:
  ""
  Returns:
      color (H*W, 3)
  """
  device = rays_o.device
    # r = td + o  compute t
  t  =  repeat((torch.linspace(near, far, bins , device = device) ), 'b -> r b' ,r = rays_o.shape[0]) #[rays , bins ]


  t_diff = t[:, 1:] - t[:, :-1]
  infinity_tensor = repeat(torch.tensor([1e10], device=device), '() -> r 1 ', r=rays_o.shape[0])
  delta = torch.cat((t_diff, infinity_tensor), dim=-1)

  #TODO add random sampling over each bin interval

  #r = [rays , 3] * [rays, bins] + [rays,3 ]
  r = rays_o[:, None, :] + t[:, :, None] * rays_d[:, None,: ] # [rays , bins , 3 ]


  c , sigma = model.intersect( rearrange(r , 'r b c -> (r b) c ') ,
                             rearrange(repeat((rays_d) , 'r c -> r b c ' , b = bins) , 'r b c -> (r b) c') ) #[rays * bins , c ]


  alpha = 1 - torch.exp(-rearrange(sigma , '(r b) 1 -> r (b 1)' , b = bins )*delta) # [rays , bins ]


  T = compute_accumulated_transmittance(1 - alpha)
  W = T.unsqueeze(2) * alpha.unsqueeze(2)
  c_hat = (W*rearrange(c , '(r b) c -> r b c' , b = bins)).sum(dim =1)
  weight_sum = W.sum(-1).sum(-1)  # Regularization for white background
  return c_hat+ 1 - weight_sum.unsqueeze(-1)
