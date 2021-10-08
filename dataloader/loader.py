from torch.utils.data import Dataset 
import numpy as np
import cv2
from util.util import uvd2xyz, xyz2uvd
from scipy.sparse import coo_matrix

class Loader(Dataset):

    def __init__(self, root, phase, img_size, dataset_name):
        assert phase in ['train', 'test']
        self.seed = np.random.RandomState(23455)
        self.root = root 
        self.phase = phase
        self.img_size = img_size
        self.dataset_name = dataset_name
        # randomly choose one of the augment options
        self.aug_ops = ['trans', 'scale', 'rot', None]

    def crop(self, img, center, csize, dsize):
        '''
        Crop hand region out of depth images, scales inverse to the distance of hand to camers
        :param center: center of mass, in image coordinates (u,v,d), d in mm
        :param csize: cube size, 3D crop volume in mm
        :param dsize: depth image size, resolution of cropped image, (w,h)
        :return: cropped hand depth image, transformation matrix for joints, center of mass in image coordinates
        '''
        assert len(csize) == 3
        assert len(dsize) == 2

        # calculate boundaries according to cube size and center
        # crop hand out of original depth image
        ustart, uend, vstart, vend, zstart, zend = self.center2bounds(center, csize)
        cropped = self.bounds2crop(img, ustart, uend, vstart, vend, zstart, zend)

        # resize depth image to same resolution
        w, h= (uend - ustart), (vend - vstart)
        # scale the longer side to corresponding dsize
        scale = min(dsize[0] / w, dsize[1] / h)
        size = (int(w * scale), int(h * scale))
        cropped = cv2.resize(cropped, size, interpolation=cv2.INTER_NEAREST)

        # pad another side to corresponding dsize
        res = np.zeros(dsize, dtype = np.float32) 
        ustart, vstart = (dsize - size)/2.
        uend, vend = ustart+size[0], vstart + size[1]

        res[int(vstart):int(vend), int(ustart):int(uend)] = cropped

        transmat = self.center2transmat(center, csize, dsize)

        return res, transmat

    def random_aug(self, sigma_trans=None, sigma_scale=None, sigma_rot=None):
        # create random augmentation paras
        # choose one of [trans, scale, rot, None] to augment
        if sigma_trans is None:
            sigma_trans = 35. 
        if sigma_scale is None:
            sigma_scale = 0.05
        if sigma_rot is None:
            sigma_rot = 180.
        
        aug_idx = self.seed.randint(0, len(self.aug_ops))
        aug_op = self.aug_ops[aug_idx]

        # trans parameters for x,y,z axis
        # normalization distribution of N(0, sigma_trans)
        trans = self.seed.randn(3) * sigma_trans 
        # normalization distribution of N(1, sigma_scale)
        scale = abs(1. + self.seed.randn() * sigma_scale)
        rot = self.seed.uniform(-sigma_rot, sigma_rot)

        return aug_op, trans, scale, rot

    def augment(self, img, jt_xyz, center, cube, M, aug_op, trans, scale, rot):
        depth_max = img.max()

        if aug_op == 'trans':
            img, jt_xyz, center, M = self.translate(img, jt_xyz, center, cube, M, trans, pad_value=0)
        elif aug_op == 'rot':
            img, jt_xyz = self.rotate(img, jt_xyz, center, rot, pad_value=0)
        elif aug_op == 'scale':
            img, cube, M = self.scale(img, center, cube, M, scale, pad_value=0)

        img = self.normalize(depth_max, img, center, cube)
        return img, jt_xyz, cube, center, M

    def normalize(self, depth_max, img, center, cube):
        img[img == depth_max] = center[2] + (cube[2] / 2.)
        # invalid points are assigned as bg
        img[img == 0] = center[2] + (cube[2] / 2.)

        img_min = center[2] - (cube[2] / 2.) # foreground, normalise to -1, should not be to much
        img_max = center[2] + (cube[2] / 2.)
        img = np.clip(img, img_min, img_max)
        # print('sum:', (img==img_min).sum())
        # scale depth 'sum:', values to [-1, 1]
        img -= center[2]
        img /= (cube[2] / 2.)

        return img # normalize to [-1, 1]

    def translate(self, img, jt_xyz, center, cube, M, trans, pad_value=0):
        '''
        Translate center.
        :param center: center of mass in image coordinated, (u,v,d)
        '''

        if np.allclose(trans, 0.):
            return img, jt_xyz, center, M 

        new_center = xyz2uvd(uvd2xyz(center, self.paras, self.flip) + trans, self.paras, self.flip)

        if not (np.allclose(center[2], 0.)) or np.allclose(new_center[2], 0.):
            new_M = self.center2transmat(new_center, cube, np.array(img.shape))
            # print(img[img>0].min()-1)
            img = self.recrop(img, new_center, cube, new_M, np.linalg.inv(M), img.shape, thresh_z=True, bg=pad_value, nv_val=np.min(img[img>0])-1)
        else:
            new_M = M
        
        jt_xyz = jt_xyz + uvd2xyz(center, self.paras, self.flip) - uvd2xyz(new_center, self.paras, self.flip)
        return img, jt_xyz, new_center, new_M 

        
    def recrop(self, img, center, cube, M, M_inv, dsize, thresh_z=True, bg=0., nv_val=0.):
        img = cv2.warpPerspective(img, np.dot(M, M_inv), dsize, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=float(bg))

        # img[np.isclose(img, 32000.)] = bg # outliers will appear on the edge
        img[img < nv_val] = bg # let 0 < depth < depth.min()-1 be background, avoiding outliers around hand

        if thresh_z:
            _, _, _, _, zstart, zend = self.center2bounds(center, cube)
            mask1 = np.logical_and(img < zstart, img != 0)
            mask2 = np.logical_and(img > zend, img != 0)
            img[mask1] = zstart
            img[mask2] = 0. 
        
        return img.astype(np.float32)


    def rotate(self, img, jt_xyz, center, rot, pad_value=0):
        '''
        Rotate hand in image coordinates.
        :return: rotated img, new jt_xyz, rotation angle in degree
        '''
        if np.allclose(rot, 0.):
            return img, jt_xyz, rot

        rot = np.mod(rot, 360)
        # -rot means rotate clockwisely
        rotM = cv2.getRotationMatrix2D((img.shape[1] // 2, img.shape[0] // 2), -rot, 1)

        img = cv2.warpAffine(img, rotM, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=pad_value)

        center_xyz = uvd2xyz(center, self.paras, self.flip)
        jt_uvd = xyz2uvd(jt_xyz + center_xyz, self.paras, self.flip)
        jt_uvd = self.rotate_pts(jt_uvd, center, rot)
        jt_xyz = uvd2xyz(jt_uvd, self.paras, self.flip) - center_xyz

        return img, jt_xyz 


    def scale(self, img, center, cube, M, scale, pad_value=0):
        '''
        Scale hand by applying different cube size.
        '''
        if np.allclose(scale, 1.):
            return img, cube, M 
        
        new_cube = cube * scale
        
        if not np.allclose(center[2], 0.):
            new_M = self.center2transmat(center, new_cube, np.array(img.shape))
            img = self.recrop(img, center, new_cube, new_M, np.linalg.inv(M), img.shape, bg=pad_value, nv_val=np.min(img[img>0])-1)
            # new_img = self.recrop(img, center, cube, new_M, np.linalg.inv(M), img.shape, bg=pad_value, nv_val=32000.)
        else:
            new_M = M
        
        return img, new_cube, new_M 

    def center2bounds(self, center, csize):

        ustart, vstart = center[:2] - (csize[:2] / 2.) / center[2] * self.paras[:2] + 0.5
        uend, vend= center[:2] + (csize[:2] / 2.) / center[2] * self.paras[:2] + 0.5
        zstart = center[2] - csize[2] / 2.
        zend = center[2] + csize[2] / 2.

        return int(ustart), int(uend), int(vstart), int(vend), zstart, zend
            
    def bounds2crop(self, img, ustart, uend, vstart, vend, zstart, zend, thresh_z=True, bg=0):
        '''
        Use boundaries to crop hand out of original depth image.
        :return: cropped image
        '''
        h, w = img.shape[:2]
        bbox = [max(vstart,0), min(vend,h), max(ustart,0), min(uend,w)]
        img = img[bbox[0]:bbox[1], bbox[2]:bbox[3]]
        # add pixels that are out of the image in order to keep aspect ratio
        img = np.pad(img, ((abs(vstart)-bbox[0], abs(vend)-bbox[1]),(abs(ustart)-bbox[2], abs(uend)-bbox[3])), mode='constant', constant_values=bg)

        if thresh_z:
            mask1 = np.logical_and(img < zstart, img != 0)
            mask2 = np.logical_and(img > zend, img != 0)
            img[mask1] = zstart
            img[mask2] = 0 
        
        return img 


    def center2transmat(self, center, csize, dsize):
        '''
        Calculate affine transform matrix for scale and translate from crop.
        :param dsize: organized as (w,h), cv2 img.shape (h,w,c)
        '''
        assert len(csize) == 3
        assert len(dsize) == 2

        # calculate boundaries according to cube size and center
        # crop hand out of original depth image
        ustart, uend, vstart, vend, _, _ = self.center2bounds(center, csize)

        trans1 = np.eye(3)
        trans1[0][2] = -ustart
        trans1[1][2] = -vstart

        w = (uend - ustart)
        h = (vend - vstart)
        # scale the longer side to corresponding dsize
        scale = min(dsize[0] / w, dsize[1] / h)
        size = (int(w * scale), int(h * scale))

        scale *= np.eye(3)
        scale[2][2] = 1

        # pad another side to corresponding dsize
        trans2 = np.eye(3)
        trans2[0][2] = int(np.floor(dsize[0] / 2. - size[0] / 2.))
        trans2[1][2] = int(np.floor(dsize[1] / 2. - size[1] / 2.))

        return np.dot(trans2, np.dot(scale, trans1)).astype(np.float32)

    def rotate_pts(self, pt, center, angle):
        '''
        Rotate single point clockwisely.
        '''
        alpha = angle * np.pi / 180.
        pt_rot = pt.copy()
        pt_rot[:, 0] = (pt[:, 0]-center[0]) * np.cos(alpha) - (pt[:, 1]-center[1]) * np.sin(alpha)
        pt_rot[:, 1] = (pt[:, 0]-center[0]) * np.sin(alpha) + (pt[:, 1]-center[1]) * np.cos(alpha)
        pt_rot[:, :2] += center[:2]

        return pt_rot.astype(np.float32)

    def transform_jt_uvd(self, jt_uvd, M):
        pts_trans = np.hstack([jt_uvd[:,:2], np.ones((jt_uvd.shape[0], 1))])

        pts_trans = np.dot(M, pts_trans.T).T
        pts_trans[:, :2] /= pts_trans[:, 2:]

        return np.hstack([pts_trans[:, :2], jt_uvd[:, 2:]]).astype(np.float32)

