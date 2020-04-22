import numpy as np

def transformPoints2D(pts, M):
    pts_trans = np.hstack([pts[:,:2], np.ones((pts.shape[0], 1))])

    pts_trans = np.dot(M, pts_trans.T).T
    pts_trans[:, :2] /= pts_trans[:, 2:]

    return np.hstack([pts_trans[:, :2], pts[:, 2:]]).astype(np.float32)


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



if __name__ == "__main__":
    M = np.random.randn(3,3)
    pts = np.random.randn(21,3)
    # pts = np.random.randn(4, 21, 3)
    camera = (588.03, 587.07, 320., 240.)
    center = np.array([1,2,3])
    csize = np.array([200, 200, 200])
    ustart, vstart = center[:2] - xyz2uvd(csize / 2., camera)[:2] + 0.5
    print(ustart, vstart)
    


