import torch
import numpy as np
# from config import opt


def xyz2uvd(pts, paras, flip=1):
    # paras: [fx, fy, fu, fv]
    pts_uvd = pts.copy()
    pts_uvd = pts_uvd.reshape(-1, 3)
    pts_uvd[:, 1] *= flip
    pts_uvd[:, :2] = pts_uvd[:, :2] * paras[:2] / pts_uvd[:, 2:] + paras[2:]

    return pts_uvd.reshape(pts.shape).astype(np.float32)


def uvd2xyz(pts, paras, flip=1):
    # paras: (fx, fy, fu, fv)
    pts_xyz = pts.copy()
    pts_xyz = pts_xyz.reshape(-1, 3)
    pts_xyz[:, :2] = (pts_xyz[:, :2] - paras[2:]) * pts_xyz[:, 2:] / paras[:2]
    pts_xyz[:, 1] *= flip

    return pts_xyz.reshape(pts.shape).astype(np.float32)


# def get_trans_points(jt, M):
#     jt_trans = torch.zeros_like(jt)
#     jt_mat = torch.cat((jt[:, :, :2], torch.ones(jt.size(0), jt.size(1), 1)), dim=-1)
#     jt_trans[:, :, 0:2] = torch.matmul(M, jt_mat.unsqueeze(-1)).squeeze(-1)[:, :, :2]
#     jt_trans[:, :, 2] = jt[:, :, 2]
#     return jt_trans
#
#
# @torch.no_grad()
# def uvd2error(joint_uvd, xyz_gt, center, M, cube, input_size, paras, flip):
#     batch_size, joint_num, _ = joint_uvd.size()
#     cube = cube.view(batch_size, 1, 3).repeat(1, joint_num, 1)
#     center = center.view(batch_size, 1, 3).repeat(1, joint_num, 1)
#     M = M.view(batch_size, 1, 3, 3).repeat(1, joint_num, 1, 1)
#     M_inv = torch.inverse(M)
#
#     joint_uvd[:, :, :2] = (joint_uvd[:, :, :2] + 1) * (input_size / 2.)
#     joint_uvd[:, :, 2] = (joint_uvd[:, :, 2]) * (cube[:, :, 2] / 2.) + center[:, :, 2]
#     joint_uvd = get_trans_points(joint_uvd, M_inv)
#     joint_xyz = uvd2xyz(joint_uvd, paras, flip)
#
#     xyz_gt = xyz_gt * cube / 2 + center
#
#     temp = (joint_xyz - xyz_gt) * (joint_xyz - xyz_gt)
#     error = torch.mean(torch.sqrt(torch.sum(torch.pow((joint_xyz-xyz_gt), 2), dim=2)))
#
#     return error, joint_uvd


if __name__ == "__main__":
    M = np.random.randn(3,3)
    pts = np.random.randn(21,3)
    # pts = np.random.randn(4, 21, 3)
    camera = (588.03, 587.07, 320., 240.)
    center = np.array([1,2,3])
    csize = np.array([200, 200, 200])
    ustart, vstart = center[:2] - xyz2uvd(csize / 2., camera)[:2] + 0.5
    print(ustart, vstart)

