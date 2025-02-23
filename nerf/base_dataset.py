import torch
from torch.utils.data.dataset import Dataset 
import numpy as np


class BaseDataset(Dataset):
    def __init__(self) -> None:
        super().__init__()


    def ndc_rays(self, H, W, focal, near, rays_o, rays_d):

        # Shift ray origins to near plane
        t = -(near + rays_o[...,2]) / rays_d[...,2]
        rays_o = rays_o + t[...,None] * rays_d
        
        # Projection
        o0 = -1./(W/(2.*focal)) * rays_o[...,0] / rays_o[...,2]
        o1 = -1./(H/(2.*focal)) * rays_o[...,1] / rays_o[...,2]
        o2 = 1. + 2. * near / rays_o[...,2]

        d0 = -1./(W/(2.*focal)) * (rays_d[...,0]/rays_d[...,2] - rays_o[...,0]/rays_o[...,2])
        d1 = -1./(H/(2.*focal)) * (rays_d[...,1]/rays_d[...,2] - rays_o[...,1]/rays_o[...,2])
        d2 = -2. * near / rays_o[...,2]
        
        rays_o = torch.stack([o0,o1,o2], -1)
        rays_d = torch.stack([d0,d1,d2], -1)
        
        return rays_o, rays_d


    def get_rays(self, H, W, focal, c2w):
        i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H))  # pytorch's meshgrid has indexing='ij'
        i = i.t()
        j = j.t()
        dirs = torch.stack([(i-W*0.5)/focal, -(j-H*0.5)/focal, -torch.ones_like(i)], -1)
        # Rotate ray directions from camera frame to the world frame
        rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
        # Translate camera frame's origin to the world frame. It is the origin of all rays.
        rays_o = c2w[:3,-1].expand(rays_d.shape)

        return rays_o, rays_d