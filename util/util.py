import torch
import numpy as np
from config import opt
from dataloader.loader import uvd2xyz

def get_trans_points(jt, M):
    jt_trans = torch.zeros_like(jt)
    jt_mat = torch.cat((jt[:, :, :2], torch.ones(jt.size(0), jt.size(1), 1)), dim=-1)
    jt_trans[:, :, 0:2] = torch.matmul(M, jt_mat.unsqueeze(-1)).squeeze(-1)[:, :, :2]
    jt_trans[:, :, 2] = jt[:, :, 2]
    return jt_trans

@torch.no_grad()
def uvd2error(joint_uvd, xyz_gt, center, M, cube, input_size, paras, flip):
    batch_size, joint_num, _ = joint_uvd.size()
    cube = cube.view(batch_size, 1, 3).repeat(1, joint_num, 1)
    center = center.view(batch_size, 1, 3).repeat(1, joint_num, 1)
    M = M.view(batch_size, 1, 3, 3).repeat(1, joint_num, 1, 1)
    M_inv = torch.inverse(M)

    joint_uvd[:, :, :2] = (joint_uvd[:, :, :2] + 1) * (input_size / 2.)
    joint_uvd[:, :, 2] = (joint_uvd[:, :, 2]) * (cube[:, :, 2] / 2.) + center[:, :, 2]
    joint_uvd = get_trans_points(joint_uvd, M_inv)
    joint_xyz = uvd2xyz(joint_uvd, paras, flip)

    xyz_gt = xyz_gt * cube / 2 + center

    temp = (joint_xyz - xyz_gt) * (joint_xyz - xyz_gt)
    error = torch.mean(torch.sqrt(torch.sum(torch.pow((joint_xyz-xyz_gt), 2), dim=2)))

    return error, joint_uvd


# def world2pixel(x, y, z, paras, flip):
#     fx, fy, u0, v0 = paras
#     u = x * fx / z + u0
#     v = flip * y * fy / z + v0
#     res = np.concatenate([u[:, np.newaxis], v[:, np.newaxis], z[:, np.newaxis]], axis=1)
#     if isinstance(x, torch.Tensor):
#         res = torch.Tensor(res).to(x.device)

#     return res

# def pixel2world(u, v, d, paras, flip):
#     fx, fy, u0, v0 = paras
#     x = (u - u0) * d / fx
#     y = flip * (v - v0) * d / fy
#     res = np.concatenate([x[:, np.newaxis], y[:, np.newaxis], d[:, np.newaxis]], axis=1)
#     if isinstance(x, torch.Tensor):
#         res = torch.Tensor(res).to(x.device)

#     return res




