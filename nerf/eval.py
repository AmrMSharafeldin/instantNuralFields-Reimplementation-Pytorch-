import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.nn import functional as F
import einops
from einops import rearrange, reduce, repeat
from tqdm import tqdm
from nerf.volume_renderer import volume_renderer
from nerf.utils import mse , mse2psnr , pose_spherical
import imageio



@torch.no_grad()
def test(model, o, d, near, far, nb_bins=100, chunk_size=20, H=400, W=400, target=None):

    o = o.chunk(chunk_size)
    d = d.chunk(chunk_size)

    image = []
    for o_batch, d_batch in zip(o, d):
        img_batch = volume_renderer(model, o_batch, d_batch, near, far, bins=nb_bins)
        image.append(img_batch) # N, 3
    image = torch.cat(image)
    image = image.reshape(H, W, 3).cpu().numpy()
    # Clip the values of the image
    image = np.clip(image, 0., 1.)
    if target is not None:
        loss = mse(image, target)
        psnr = mse2psnr(loss)

    if target is not None:
        return image,mse, psnr
    else:
        return image
    



def eval(model , test_o , test_d , H=400, W=400 , near = 8 , far = 12 , video_path = 'novel_views360.mp4' ) :

  frames = []

  device = 'cuda' if torch.cuda.is_available() else 'cpu'

  for theta in range(0, 360, 5):  # Increment by 5 degrees for smoother animation
      c2w = pose_spherical(theta, 0.0)

      rays_o_rot = (c2w[:3, :3] @ test_o.T).T + c2w[:3, 3]
      rays_d_rot = (c2w[:3, :3] @ test_d.T).T

      with torch.no_grad():
          model.eval()
          img = test(model, torch.from_numpy(rays_o_rot).to(model.device).float(),
                    torch.from_numpy(rays_d_rot).to(model.device).float(), near = near , far = far)

      # Convert to uint8 image format
      img = (img.clip(0, 1) * 255).astype(np.uint8)
      frames.append(img)
      #plt.imshow(img)
      #plt.show()
  # Save frames to a video
  imageio.mimsave(video_path, frames, fps=20)

  print(f"Video saved to {video_path}")