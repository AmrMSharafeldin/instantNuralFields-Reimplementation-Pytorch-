import numpy as np
import matplotlib.pyplot as plt
import torch
import einops
from einops import rearrange, reduce, repeat
import os
import imageio
from torch.utils.data import DataLoader



def load_custom_data(datapath, mode='train'):

    pose_file_names = [f for f in os.listdir(datapath + f'/{mode}/pose') if f.endswith('.txt')]
    intrisics_file_names = [f for f in os.listdir(datapath + f'/{mode}/intrinsics') if f.endswith('.txt')]
    img_file_names = [f for f in os.listdir(datapath + '/imgs') if mode in f]

    assert len(pose_file_names) == len(intrisics_file_names)
    assert len(img_file_names) == len(pose_file_names)

    # Read
    N = len(pose_file_names)
    poses = np.zeros((N, 4, 4))
    intrinsics = np.zeros((N, 4, 4))

    images = []

    for i in range(N):
        name = pose_file_names[i]

        pose = open(datapath + f'/{mode}/pose/' + name).read().split()
        poses[i] = np.array(pose, dtype=float).reshape(4, 4)

        intrinsic = open(datapath + f'/{mode}/intrinsics/' + name).read().split()
        intrinsics[i] = np.array(intrinsic, dtype=float).reshape(4, 4)

        # Read images
        img = imageio.imread(datapath + '/imgs/' + name.replace('txt', 'png')) / 255.
        images.append(img[None, ...])
    images = np.concatenate(images)

    H = images.shape[1]
    W = images.shape[2]

    if images.shape[3] == 4: #RGBA -> RGB
        images = images[..., :3] * images[..., -1:] + (1 - images[..., -1:])

    rays_o = np.zeros((N, H*W, 3))
    rays_d = np.zeros((N, H*W, 3))
    target_px_values = images.reshape((N, H*W, 3))

    for i in range(N):

        c2w = poses[i]
        f = intrinsics[i, 0, 0]

        u = np.arange(W)
        v = np.arange(H)
        u, v = np.meshgrid(u, v)
        dirs = np.stack((u - W / 2, -(v - H / 2), - np.ones_like(u) * f), axis=-1)
        dirs = (c2w[:3, :3] @ dirs[..., None]).squeeze(-1)
        dirs = dirs / np.linalg.norm(dirs, axis=-1, keepdims=True)

        rays_d[i] = dirs.reshape(-1, 3)
        rays_o[i] += c2w[:3, 3]

    return rays_o, rays_d, target_px_values





def trilinear_interpolation(res, grid, points, grid_type):
    """
    Performs trilinear interpolation of points with respect to a grid.

    Parameters:
        res (int): Resolution of the grid in each dimension.
        grid (torch.Tensor): A 3D torch tensor representing the grid.
        points (torch.Tensor): A 2D torch tensor of shape (n, 3) representing
            the points to interpolate.
        grid_type (str): Type of grid.

    Returns:
        torch.Tensor: A 1D torch tensor of shape (n,) representing the interpolated
            values at the given points.
    """
    PRIMES = [1, 265443567, 805459861]

    # Get the dimensions of the grid
    grid_size, feat_size = grid.shape
    points = points.unsqueeze(0)
    _, N, _ = points.shape

    # Get the x, y, and z coordinates of the eight nearest points for each input point
    x = points[:, :, 0] * (res - 1)
    y = points[:, :, 1] * (res - 1)
    z = points[:, :, 2] * (res - 1)

    x1 = torch.floor(torch.clip(x, 0, res - 1 - 1e-5)).int()
    y1 = torch.floor(torch.clip(y, 0, res - 1 - 1e-5)).int()
    z1 = torch.floor(torch.clip(z, 0, res - 1 - 1e-5)).int()

    x2 = torch.clip(x1 + 1, 0, res - 1).int()
    y2 = torch.clip(y1 + 1, 0, res - 1).int()
    z2 = torch.clip(z1 + 1, 0, res - 1).int()

    # Compute the weights for each of the eight points
    w1 = (x2 - x) * (y2 - y) * (z2 - z)
    w2 = (x - x1) * (y2 - y) * (z2 - z)
    w3 = (x2 - x) * (y - y1) * (z2 - z)
    w4 = (x - x1) * (y - y1) * (z2 - z)
    w5 = (x2 - x) * (y2 - y) * (z - z1)
    w6 = (x - x1) * (y2 - y) * (z - z1)
    w7 = (x2 - x) * (y - y1) * (z - z1)
    w8 = (x - x1) * (y - y1) * (z - z1)

    if grid_type == "NGLOD":
        # Interpolate the values for each point
        id1 = (x1 + y1 * res + z1 * res * res).long()
        id2 = (x2 + y1 * res + z1 * res * res).long()
        id3 = (x1 + y2 * res + z1 * res * res).long()
        id4 = (x2 + y2 * res + z1 * res * res).long()
        id5 = (x1 + y1 * res + z2 * res * res).long()
        id6 = (x2 + y1 * res + z2 * res * res).long()
        id7 = (x1 + y2 * res + z2 * res * res).long()
        id8 = (x2 + y2 * res + z2 * res * res).long()

    elif grid_type == "HASH":
        npts = res**3
        if npts > grid_size:
            id1 = ((x1 * PRIMES[0]) ^ (y1 * PRIMES[1]) ^ (z1 * PRIMES[2])) % grid_size
            id2 = ((x2 * PRIMES[0]) ^ (y1 * PRIMES[1]) ^ (z1 * PRIMES[2])) % grid_size
            id3 = ((x1 * PRIMES[0]) ^ (y2 * PRIMES[1]) ^ (z1 * PRIMES[2])) % grid_size
            id4 = ((x2 * PRIMES[0]) ^ (y2 * PRIMES[1]) ^ (z1 * PRIMES[2])) % grid_size
            id5 = ((x1 * PRIMES[0]) ^ (y1 * PRIMES[1]) ^ (z2 * PRIMES[2])) % grid_size
            id6 = ((x2 * PRIMES[0]) ^ (y1 * PRIMES[1]) ^ (z2 * PRIMES[2])) % grid_size
            id7 = ((x1 * PRIMES[0]) ^ (y2 * PRIMES[1]) ^ (z2 * PRIMES[2])) % grid_size
            id8 = ((x2 * PRIMES[0]) ^ (y2 * PRIMES[1]) ^ (z2 * PRIMES[2])) % grid_size
        else:
            id1 = (x1 + y1 * res + z1 * res * res).long()
            id2 = (x2 + y1 * res + z1 * res * res).long()
            id3 = (x1 + y2 * res + z1 * res * res).long()
            id4 = (x2 + y2 * res + z1 * res * res).long()
            id5 = (x1 + y1 * res + z2 * res * res).long()
            id6 = (x2 + y1 * res + z2 * res * res).long()
            id7 = (x1 + y2 * res + z2 * res * res).long()
            id8 = (x2 + y2 * res + z2 * res * res).long()
    else:
        print("NOT IMPLEMENTED")

    values = (
        torch.einsum("ab,abc->abc", w1, grid[(id1).long()])
        + torch.einsum("ab,abc->abc", w2, grid[(id2).long()])
        + torch.einsum("ab,abc->abc", w3, grid[(id3).long()])
        + torch.einsum("ab,abc->abc", w4, grid[(id4).long()])
        + torch.einsum("ab,abc->abc", w5, grid[(id5).long()])
        + torch.einsum("ab,abc->abc", w6, grid[(id6).long()])
        + torch.einsum("ab,abc->abc", w7, grid[(id7).long()])
        + torch.einsum("ab,abc->abc", w8, grid[(id8).long()])
    )
    return values[0]




def mse(prediction, target):
    return ((prediction - target) ** 2).mean()

def mse2psnr(mse):
    return 20 * np.log10(1 / np.sqrt(mse))



def pose_spherical(theta, radius):
    trans_t = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ], dtype=np.float32)

    rot_theta = np.array([
        [np.cos(np.radians(theta)), -np.sin(np.radians(theta)), 0, 0],
        [np.sin(np.radians(theta)), np.cos(np.radians(theta)), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ], dtype=np.float32)

    c2w = trans_t @ rot_theta

    return c2w






def setup_data_loaders(data_path, batch_size , custom = True , data_set_train = None , data_set_test = None):
    # Load training and warmup data
    if custom:
        o, d, target_px_values = load_custom_data(data_path, mode='train')

        # Load test data
        test_o, test_d, test_px = load_custom_data(data_path, mode='test')
    else : 
        o , d , target_px_values = data_set_train
        test_o, test_d, test_px = data_set_test
    # Create DataLoader for training
    train_data = torch.cat((
        torch.from_numpy(o).reshape(-1, 3).float(),
        torch.from_numpy(d).reshape(-1, 3).float(),
        torch.from_numpy(target_px_values).reshape(-1, 3).float()
    ), dim=1)
    
    dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    # Create DataLoader for warmup
    o_warmup = torch.from_numpy(o).reshape(90, 400, 400, 3)[:, 100:300, 100:300, :].reshape(-1, 3).float()
    d_warmup = torch.from_numpy(d).reshape(90, 400, 400, 3)[:, 100:300, 100:300, :].reshape(-1, 3).float()
    target_px_values_warmup = torch.from_numpy(target_px_values).reshape(90, 400, 400, 3)[:, 100:300, 100:300, :].reshape(-1, 3).float()
    
    warmup_data = torch.cat((
        o_warmup,
        d_warmup,
        target_px_values_warmup
    ), dim=1)
    
    dataloader_warmup = DataLoader(warmup_data, batch_size=batch_size, shuffle=True)
    

    
    return dataloader, dataloader_warmup, (test_o, test_d, test_px)
