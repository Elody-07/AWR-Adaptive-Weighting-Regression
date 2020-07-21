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

    def transform_jt_uvd(self, jt_uvd, M):
        pts_trans = np.hstack([jt_uvd[:,:2], np.ones((jt_uvd.shape[0], 1))])

        pts_trans = np.dot(M, pts_trans.T).T
        pts_trans[:, :2] /= pts_trans[:, 2:]

        return np.hstack([pts_trans[:, :2], jt_uvd[:, 2:]]).astype(np.float32)

