# -*- coding: utf-8 -*-
from torch.utils.data import Dataset
import torch
import torch.nn.functional as F
import numpy as np
import scipy.io as sio
import math
import cv2
import sys
import os
from PIL import Image
from torch.utils.data import DataLoader
import h5py
import time
import sys
sys.path.append('..')
# import generateFeature as G
# from vis_tool import *
from sklearn.decomposition import PCA


joint_select = np.array([0,1,3,5,  6,7,9,11,  12,13,15,17,  18,19,21,23,  24,25,27,28,  32,30,31])
calculate = [0,2,4,6,8,10,12,14,16,17,18,21,22,20]

xrange = range

def get_center_adopt(img):
    img_dim = img.reshape(-1)
    min_value = np.sort(img_dim)[:1000].mean()
    centers = np.array([0.0, 0.0, 300.0])
    flag = np.logical_and(img <= min_value + 100, img >= min_value)
    x = np.linspace(0, img.shape[1], img.shape[1])
    y = np.linspace(0, img.shape[0], img.shape[0])
    xv, yv = np.meshgrid(x, y)
    centers[0] = np.mean(xv[flag])
    centers[1] = np.mean(yv[flag])
    centers[2] = np.mean(img[flag])
    if centers[2] <= 0:
        centers[2] = 300.0
    if not flag.any():
        centers[0] = 0
        centers[1] = 0
        centers[2] = 300.0
    return centers


def get_center_fast(img, upper=650, lower=100):
    centers = np.array([0.0, 0.0, 300.0])
    flag = np.logical_and(img <= upper, img >= lower)
    x = np.linspace(0, img.shape[1], img.shape[1])
    y = np.linspace(0, img.shape[0], img.shape[0])
    xv, yv = np.meshgrid(x, y)
    centers[0] = np.mean(xv[flag])
    centers[1] = np.mean(yv[flag])
    centers[2] = np.mean(img[flag])
    if centers[2] <= 0:
        centers[2] = 300.0
    if not flag.any():
        centers[0] = 0
        centers[1] = 0
        centers[2] = 300.0
    return centers


def pixel2world(x, y, z, paras):
    fx,fy,fu,fv = paras
    worldX = (x - fu) * z / fx
    worldY = (fv - y) * z / fy
    return worldX, worldY


def pixel2world_noflip(x, y, z, paras):
    fx,fy,fu,fv = paras
    worldX = (x - fu) * z / fx
    worldY = (y - fv) * z / fy
    return worldX, worldY


def world2pixel(x, y, z, paras):
    fx, fy, fu, fv = paras
    pixelX = x * fx / z + fu
    pixelY = fv - y * fy / z
    return pixelX, pixelY


def rotatePoint2D(p1, center, angle):
    """
    Rotate a point in 2D around center
    :param p1: point in 2D (u,v,d)
    :param center: 2D center of rotation
    :param angle: angle in deg
    :return: rotated point
    """
    alpha = angle * np.pi / 180.
    pp = p1.copy()
    pp[0:2] -= center[0:2]
    pr = np.zeros_like(pp)
    pr[0] = pp[0] * np.cos(alpha) - pp[1] * np.sin(alpha)
    pr[1] = pp[0] * np.sin(alpha) + pp[1] * np.cos(alpha)
    pr[2] = pp[2]
    ps = pr
    ps[0:2] += center[0:2]
    return ps


def rotatePoints2D(pts, center, angle):
    """
    Transform points in 2D coordinates
    :param pts: point coordinates
    :param center: 2D center of rotation
    :param angle: angle in deg
    :return: rotated points
    """
    ret = pts.copy()
    for i in range(pts.shape[0]):
        ret[i] = rotatePoint2D(pts[i], center, angle)
    return ret


def transformPoint2D(pt, M):
    """
    Transform point in 2D coordinates
    :param pt: point coordinates
    :param M: transformation matrix
    :return: transformed point
    """
    pt2 = np.dot(np.asarray(M).reshape((3, 3)), np.asarray([pt[0], pt[1], 1]))
    return np.asarray([pt2[0] / pt2[2], pt2[1] / pt2[2]])


def transformPoints2D(pts, M):
    """
    Transform points in 2D coordinates
    :param pts: point coordinates
    :param M: transformation matrix
    :return: transformed points
    """
    ret = pts.copy()
    for i in range(pts.shape[0]):
        ret[i, 0:2] = transformPoint2D(pts[i, 0:2], M)
    return ret


# use opencv, if will change the depth value
def Crop_Image_ren(depth, center, cube_size, paras, croppedSz):
    fx, fy, fu, fv = paras
    u = center[0]
    v = center[1]
    d = center[2]

    zstart = d - cube_size / 2.
    zend = d + cube_size / 2.
    xstart = int(math.floor((u * d / fx - cube_size / 2.) / d * fx))
    xend = int(math.floor((u * d / fx + cube_size / 2.) / d * fx))
    ystart = int(math.floor((v * d / fy - cube_size / 2.) / d * fy))
    yend = int(math.floor((v * d / fy + cube_size / 2.) / d * fy))

    src = np.float32([(xstart, ystart), (xstart, yend), (xend, ystart)])
    dst = np.float32([(0, 0), (0, croppedSz - 1), (croppedSz - 1, 0)])
    trans = cv2.getAffineTransform(np.array(src, dtype=np.float32),
                np.array(dst, dtype=np.float32))
    res_img = cv2.warpAffine(depth, trans, (croppedSz, croppedSz), None,
                cv2.INTER_LINEAR, cv2.BORDER_REPLICATE)#borderValue

    msk1 = np.logical_and(res_img < zstart, res_img != 0)
    msk2 = np.logical_and(res_img > zend, res_img != 0)
    res_img[msk1] = zstart
    res_img[msk2] = 0.  # backface is at 0, it is set later
    return res_img, trans


def getCrop(depth, xstart, xend, ystart, yend, zstart, zend, thresh_z=True, background=0):
    """
    Crop patch from image
    :param depth: depth image to crop from
    :param xstart: start x
    :param xend: end x
    :param ystart: start y
    :param yend: end y
    :param zstart: start z
    :param zend: end z
    :param thresh_z: threshold z values
    :return: cropped image
    """
    if len(depth.shape) == 2:
        cropped = depth[max(ystart, 0):min(yend, depth.shape[0]), max(xstart, 0):min(xend, depth.shape[1])].copy()
        # add pixels that are out of the image in order to keep aspect ratio
        cropped = np.pad(cropped, ((abs(ystart)-max(ystart, 0),
                                       abs(yend)-min(yend, depth.shape[0])),
                                      (abs(xstart)-max(xstart, 0),
                                       abs(xend)-min(xend, depth.shape[1]))), mode='constant', constant_values=background)
    elif len(depth.shape) == 3:
        cropped = depth[max(ystart, 0):min(yend, depth.shape[0]), max(xstart, 0):min(xend, depth.shape[1]), :].copy()
        # add pixels that are out of the image in order to keep aspect ratio
        cropped = np.pad(cropped, ((abs(ystart)-max(ystart, 0),
                                       abs(yend)-min(yend, depth.shape[0])),
                                      (abs(xstart)-max(xstart, 0),
                                       abs(xend)-min(xend, depth.shape[1])),
                                      (0, 0)), mode='constant', constant_values=background)
    else:
        raise NotImplementedError()

    if thresh_z is True:
        msk1 = np.logical_and(cropped < zstart, cropped != 0)
        msk2 = np.logical_and(cropped > zend, cropped != 0)
        cropped[msk1] = zstart
        cropped[msk2] = 0.  # backface is at 0, it is set later
    return cropped


def nyu_reader(img_path):
    img = cv2.imread(img_path)
    depth = np.asarray(img[:, :, 0] + img[:, :, 1] * 256, dtype=np.float32)
    return depth


def icvl_reader(img_path):
    img = Image.open(img_path)  # open image
    assert len(img.getbands()) == 1  # ensure depth image
    depth = np.asarray(img, np.float32)
    return depth


def msra_reader(image_name,para):
    f = open(image_name, 'rb')
    data = np.fromfile(f, dtype=np.uint32)
    width, height, left, top, right , bottom = data[:6]
    depth = np.zeros((height, width), dtype=np.float32)
    f.seek(4*6)
    data = np.fromfile(f, dtype=np.float32)
    depth[top:bottom, left:right] = np.reshape(data, (bottom-top, right-left))
    depth_pcl = np.reshape(data, (bottom-top, right-left))
    #convert to world
    imgHeight, imgWidth = depth_pcl.shape
    hand_3d = np.zeros([3, imgHeight*imgWidth])
    d2Output_x = np.tile(np.arange(imgWidth), (imgHeight, 1)).reshape(imgHeight, imgWidth).astype('float64') + left
    d2Output_y = np.repeat(np.arange(imgHeight), imgWidth).reshape(imgHeight, imgWidth).astype('float64') + top
    hand_3d[0], hand_3d[1] = pixel2world(d2Output_x.reshape(-1), d2Output_y.reshape(-1), depth_pcl.reshape(-1),para)
    hand_3d[2] = depth_pcl.reshape(-1)
    valid = np.arange(0,imgWidth*imgHeight)
    valid = valid[(hand_3d[0, :] != 0)|(hand_3d[1, :] != 0)|(hand_3d[2, :] != 0)]
    handpoints = hand_3d[:, valid].transpose(1,0)

    return depth,handpoints


def hands17_reader(img_path):
    img = Image.open(img_path)  # open image
    assert len(img.getbands()) == 1  # ensure depth image
    depth = np.asarray(img, np.float32)
    return depth


def get_center(img, upper=1000, lower=10):
    centers = np.array([0.0, 0.0, 300.0])
    flag = np.logical_and(img <= upper, img >= lower)
    x = np.linspace(0, img.shape[1], img.shape[1])
    y = np.linspace(0, img.shape[0], img.shape[0])
    xv, yv = np.meshgrid(x, y)
    centers[0] = np.mean(xv[flag])
    centers[1] = np.mean(yv[flag])
    centers[2] = np.mean(img[flag])
    if centers[2] <= 0:
        centers[2] = 300.0
    if not flag.any():
        centers[0] = 0
        centers[1] = 0
        centers[2] = 300.0
    return centers


class loader(Dataset):
    def __init__(self,root_dir, img_type, img_size, center_type,dataset_name):
        self.rng = np.random.RandomState(23455)
        self.dataset_name = dataset_name
        self.root_dir = root_dir
        self.img_type = img_type
        self.img_size = img_size
        self.center_type = center_type
        self.allJoints = False
        # create OBB
        self.pca = PCA(n_components=3)
        self.sample_num = 1024

    # joint position from image to world
    def jointImgTo3D(self, uvd):
        fx, fy, fu, fv = self. paras
        ret = np.zeros_like(uvd, np.float32)
        if len(ret.shape) == 1:
            ret[0] = (uvd[0] - fu) * uvd[2] / fx
            ret[1] = self.flip * (uvd[1] - fv) * uvd[2] / fy
            ret[2] = uvd[2]
        elif len(ret.shape) == 2:
            ret[:, 0] = (uvd[:, 0] - fu) * uvd[:, 2] / fx
            ret[:, 1] = self.flip * (uvd[:, 1] - fv) * uvd[:, 2] / fy
            ret[:, 2] = uvd[:, 2]
        else:
            ret[:, :, 0] = (uvd[:, :, 0] - fu) * uvd[:, :, 2] / fx
            ret[:, :, 1] = self.flip * (uvd[:, :, 1] - fv) * uvd[:, :, 2] / fy
            ret[:, :, 2] = uvd[:, :, 2]

        return ret

    def joint3DToImg(self,xyz):
        fx, fy, fu, fv = self.paras
        ret = np.zeros_like(xyz, np.float32)
        if len(ret.shape) == 1:
            ret[0] = (xyz[0] * fx / xyz[2] + fu)
            ret[1] = (self.flip * xyz[1] * fy / xyz[2] + fv)
            ret[2] = xyz[2]
        elif len(ret.shape) == 2:
            ret[:, 0] = (xyz[:, 0] * fx / xyz[:, 2] + fu)
            ret[:, 1] = (self.flip * xyz[:, 1] * fy / xyz[:, 2] + fv)
            ret[:, 2] = xyz[:, 2]
        else:
            ret[:, :, 0] = (xyz[:, :, 0] * fx / xyz[:, :, 2] + fu)
            ret[:, :, 1] = (self.flip * xyz[:, :, 1] * fy / xyz[:, :, 2] + fv)
            ret[:, :, 2] = xyz[:, :, 2]
        return ret

    def comToBounds(self, com, size):
        fx, fy, fu, fv = self.paras
        zstart = com[2] - size[2] / 2.
        zend = com[2] + size[2] / 2.
        xstart = int(np.floor((com[0] * com[2] / fx - size[0] / 2.) / com[2] * fx + 0.5))
        xend = int(np.floor((com[0] * com[2] / fx + size[0] / 2.) / com[2] * fx + 0.5))
        ystart = int(np.floor((com[1] * com[2] / fy - size[1] / 2.) / com[2] * fy + 0.5))
        yend = int(np.floor((com[1] * com[2] / fy + size[1] / 2.) / com[2] * fy + 0.5))
        return xstart, xend, ystart, yend, zstart, zend

    def comToTransform(self, com, size, dsize):
        """
        Calculate affine transform from crop
        :param com: center of mass, in image coordinates (x,y,z), z in mm
        :param size: (x,y,z) extent of the source crop volume in mm
        :return: affine transform
        """

        xstart, xend, ystart, yend, _, _ = self.comToBounds(com, size)

        trans = np.eye(3)
        trans[0, 2] = -xstart
        trans[1, 2] = -ystart

        wb = (xend - xstart)
        hb = (yend - ystart)
        if wb > hb:
            scale = np.eye(3) * dsize[0] / float(wb)
            sz = (dsize[0], hb * dsize[0] / wb)
        else:
            scale = np.eye(3) * dsize[1] / float(hb)
            sz = (wb * dsize[1] / hb, dsize[1])
        scale[2, 2] = 1

        # ori
        # xstart = int(np.floor(dsize[0] / 2. - sz[1] / 2.))
        # ystart = int(np.floor(dsize[1] / 2. - sz[0] / 2.))

        # change by pengfeiren
        xstart = int(np.floor(dsize[0] / 2. - sz[0] / 2.))
        ystart = int(np.floor(dsize[1] / 2. - sz[1] / 2.))
        off = np.eye(3)
        off[0, 2] = xstart
        off[1, 2] = ystart

        return np.dot(off, np.dot(scale, trans))

    def recropHand(self, crop, M, Mnew, target_size, background_value=0., nv_val=0., thresh_z=True, com=None,
                   size=(250, 250, 250)):

        flags = cv2.INTER_LINEAR

        warped = cv2.warpPerspective(crop, np.dot(M, Mnew), target_size, flags=flags,
                                     borderMode=cv2.BORDER_CONSTANT, borderValue=float(background_value))
        warped[np.isclose(warped, nv_val)] = background_value

        if thresh_z is True:
            assert com is not None
            _, _, _, _, zstart, zend = self.comToBounds(com, size)
            msk1 = np.logical_and(warped < zstart, warped != 0)
            msk2 = np.logical_and(warped > zend, warped != 0)
            warped[msk1] = zstart
            warped[msk2] = 0.  # backface is at 0, it is set later

        return warped

    def moveCoM(self, dpt, cube, com, off, joints3D, M, pad_value=0):
        """
        Adjust already cropped image such that a moving CoM normalization is simulated
        :param dpt: cropped depth image with different CoM
        :param cube: metric cube of size (sx,sy,sz)
        :param com: original center of mass, in image coordinates (x,y,z)
        :param off: offset to center of mass (dx,dy,dz) in 3D coordinates
        :param joints3D: 3D joint coordinates, cropped to old CoM
        :param pad_value: value of padding
        :return: adjusted image, new 3D joint coordinates, new center of mass in image coordinates
        """

        # if offset is 0, nothing to do
        if np.allclose(off, 0.):
            return dpt, joints3D, com, M

        # add offset to com
        new_com = self.joint3DToImg(self.jointImgTo3D(com) + off)

        # check for 1/0.
        if not (np.allclose(com[2], 0.) or np.allclose(new_com[2], 0.)):
            # scale to original size
            Mnew = self.comToTransform(new_com, cube, dpt.shape)
            new_dpt = self.recropHand(dpt, Mnew, np.linalg.inv(M), dpt.shape, background_value=pad_value,
                                 nv_val=32000., thresh_z=True, com=new_com, size=cube)
        else:
            Mnew = M
            new_dpt = dpt

        # adjust joint positions to new CoM
        new_joints3D = joints3D + self.jointImgTo3D(com) - self.jointImgTo3D(new_com)

        return new_dpt, new_joints3D, new_com, Mnew

    def rotateHand(self, dpt, cube, com, rot, joints3D, pad_value=0):
        """
        Rotate hand virtually in the image plane by a given angle
        :param dpt: cropped depth image with different CoM
        :param cube: metric cube of size (sx,sy,sz)
        :param com: original center of mass, in image coordinates (x,y,z)
        :param rot: rotation angle in deg
        :param joints3D: original joint coordinates, in 3D coordinates (x,y,z)
        :param pad_value: value of padding
        :return: adjusted image, new 3D joint coordinates, rotation angle in XXX
        """

        # if rot is 0, nothing to do
        if np.allclose(rot, 0.):
            return dpt, joints3D, rot

        rot = np.mod(rot, 360)

        M = cv2.getRotationMatrix2D((dpt.shape[1] // 2, dpt.shape[0] // 2), -rot, 1)

        flags = cv2.INTER_LINEAR

        new_dpt = cv2.warpAffine(dpt, M, (dpt.shape[1], dpt.shape[0]), flags=flags,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=pad_value)

        com3D = self.jointImgTo3D(com)
        joint_2D = self.joint3DToImg(joints3D + com3D)
        data_2D = np.zeros_like(joint_2D)
        for k in xrange(data_2D.shape[0]):
            data_2D[k] = rotatePoint2D(joint_2D[k], com[0:2], rot)
        new_joints3D = (self.jointImgTo3D(data_2D) - com3D)

        return new_dpt, new_joints3D, rot

    def scaleHand(self, dpt, cube, com, sc, joints3D, M, pad_value=0):
        """
        Virtually scale the hand by applying different cube
        :param dpt: cropped depth image with different CoM
        :param cube: metric cube of size (sx,sy,sz)
        :param com: original center of mass, in image coordinates (x,y,z)
        :param sc: scale factor for cube
        :param joints3D: 3D joint coordinates, cropped to old CoM
        :param pad_value: value of padding
        :return: adjusted image, new 3D joint coordinates, new center of mass in image coordinates
        """

        # if scale is 1, nothing to do
        if np.allclose(sc, 1.):
            return dpt, joints3D, cube, M

        new_cube = [s * sc for s in cube]

        # check for 1/0.
        if not np.allclose(com[2], 0.):
            # scale to original size
            Mnew = self.comToTransform(com, new_cube, dpt.shape)
            new_dpt = self.recropHand(dpt, Mnew, np.linalg.inv(M), dpt.shape, background_value=pad_value,
                                 nv_val=32000., thresh_z=True, com=com, size=cube)
        else:
            Mnew = M
            new_dpt = dpt

        new_joints3D = joints3D

        return new_dpt, new_joints3D, new_cube, Mnew

    def rand_augment(self, sigma_com=None, sigma_sc=None, rot_range=None):
        if sigma_com is None:
            sigma_com = 35.

        if sigma_sc is None:
            sigma_sc = 0.05

        if rot_range is None:
            rot_range = 180.

        mode = self.rng.randint(0, len(self.aug_modes))
        off = self.rng.randn(3) * sigma_com  # +-px/mm
        rot = self.rng.uniform(-rot_range, rot_range)
        sc = abs(1. + self.rng.randn() * sigma_sc)

        return mode, off, rot, sc

    def augmentCrop(self,img, gt3Dcrop, com, cube, M, mode, off, rot, sc, normZeroOne=False):
        """
        Commonly used function to augment hand poses
        :param img: image
        :param gt3Dcrop: 3D annotations
        :param com: center of mass in image coordinates (x,y,z)
        :param cube: cube
        :param aug_modes: augmentation modes
        :param hd: hand detector
        :param normZeroOne: normalization
        :param sigma_com: sigma of com noise
        :param sigma_sc: sigma of scale noise
        :param rot_range: rotation range in degrees
        :return: image, 3D annotations(unnormal), com(image coordinates), cube
        """
        assert len(img.shape) == 2
        assert isinstance(self.aug_modes, list)
        premax = img.max()
        if self.aug_modes[mode] == 'com':
            rot = 0.
            sc = 1.
            imgD, new_joints3D, com, M = self.moveCoM(img.astype('float32'), cube, com, off, gt3Dcrop, M, pad_value=0)
        elif self.aug_modes[mode] == 'rot':
            off = np.zeros((3,))
            sc = 1.
            imgD, new_joints3D, rot = self.rotateHand(img.astype('float32'), cube, com, rot, gt3Dcrop, pad_value=0)
        elif self.aug_modes[mode] == 'sc':
            off = np.zeros((3,))
            rot = 0.
            imgD, new_joints3D, cube, M = self.scaleHand(img.astype('float32'), cube, com, sc, gt3Dcrop, M, pad_value=0)
        elif self.aug_modes[mode] == 'none':
            off = np.zeros((3,))
            sc = 1.
            rot = 0.
            imgD = img
            new_joints3D = gt3Dcrop
        else:
            raise NotImplementedError()
        imgD = self.normalize_img(premax, imgD, com, cube)
        return imgD, None, new_joints3D, np.asarray(cube), com, M, rot

    def normalize_img(self, premax, imgD, com, cube):
        imgD[imgD == premax] = com[2] + (cube[2] / 2.)
        imgD[imgD == 0] = com[2] + (cube[2] / 2.)
        imgD[imgD >= com[2] + (cube[2] / 2.)] = com[2] + (cube[2] / 2.)
        imgD[imgD <= com[2] - (cube[2] / 2.)] = com[2] - (cube[2] / 2.)
        imgD -= com[2]
        imgD /= (cube[2] / 2.)
        return imgD

    # use deep-pp's method
    def Crop_Image_deep_pp(self, depth, com, size, dsize):
        """
        Crop area of hand in 3D volumina, scales inverse to the distance of hand to camera
        :param com: center of mass, in image coordinates (x,y,z), z in mm
        :param size: (x,y,z) extent of the source crop volume in mm
        :param dsize: (x,y) extent of the destination size
        :return: cropped hand image, transformation matrix for joints, CoM in image coordinates
        """

        # print com, self.importer.jointImgTo3D(com)
        # import matplotlib.pyplot as plt
        # import matplotlib
        # fig = plt.figure()
        # ax = fig.add_subplot(111)
        # ax.imshow(self.dpt, cmap=matplotlib.cm.jet)

        if len(size) != 3 or len(dsize) != 2:
            raise ValueError("Size must be 3D and dsize 2D bounding box")

        # calculate boundaries
        xstart, xend, ystart, yend, zstart, zend = self.comToBounds(com, size)

        # crop patch from source
        cropped = getCrop(depth, xstart, xend, ystart, yend, zstart, zend)

        # resize to same size
        wb = (xend - xstart)
        hb = (yend - ystart)
        if wb > hb:
            sz = (dsize[0], int(hb * dsize[0] / wb))
        else:
            sz = (int(wb * dsize[1] / hb), dsize[1])

        trans = np.eye(3)
        trans[0, 2] = -xstart
        trans[1, 2] = -ystart
        if cropped.shape[0] > cropped.shape[1]:
            scale = np.eye(3) * sz[1] / float(cropped.shape[0])
        else:
            scale = np.eye(3) * sz[0] / float(cropped.shape[1])
        scale[2, 2] = 1

        # depth resize
        rz = cv2.resize(cropped, sz, interpolation=cv2.INTER_NEAREST)

        ret = np.ones(dsize, np.float32) * 0  # use background as filler
        xstart = int(np.floor(dsize[0] / 2. - rz.shape[1] / 2.))
        xend = int(xstart + rz.shape[1])
        ystart = int(np.floor(dsize[1] / 2. - rz.shape[0] / 2.))
        yend = int(ystart + rz.shape[0])
        ret[ystart:yend, xstart:xend] = rz
        # print rz.shape, xstart, ystart
        off = np.eye(3)
        off[0, 2] = xstart
        off[1, 2] = ystart

        return ret, np.dot(off, np.dot(scale, trans))

    def pca_point(self, pcl, joint):
        point_num = pcl.shape[0]
        if point_num < 10:
            pcl =  self.joint2pc(joint)
        self.pca.fit(pcl)
        coeff = self.pca.components_.T
        if coeff[1, 0] < 0:
            coeff[:, 0] = -coeff[:, 0]
        if coeff[2, 2] < 0:
            coeff[:, 2] = -coeff[:, 2]
        coeff[:, 1] = np.cross(coeff[:, 2], coeff[:, 0])
        points_rotation = np.dot(pcl, coeff)
        joint_rotation = np.dot(joint, coeff)

        index = np.arange(points_rotation.shape[0])
        if points_rotation.shape[0] < self.sample_num:
            tmp = math.floor(self.sample_num / points_rotation.shape[0])
            index_temp = index.repeat(tmp)
            index = np.append(index_temp,np.random.choice(index, size=divmod(self.sample_num, points_rotation.shape[0])[1],
                                               replace=False))
        index = np.random.choice(index, size=self.sample_num, replace=False)
        points_rotation_sampled = points_rotation[index]

        # Normalize Point Cloud
        scale = 1.2
        bb3d_x_len = scale * (points_rotation[:, 0].max() - points_rotation[:, 0].min())
        bb3d_y_len = scale * (points_rotation[:, 1].max() - points_rotation[:, 1].min())
        bb3d_z_len = scale * (points_rotation[:, 2].max() - points_rotation[:, 2].min())
        max_bb3d_len = bb3d_x_len / 2.0

        points_rotation_sampled_normalized = points_rotation_sampled / max_bb3d_len
        joint_rotation_normalized = joint_rotation / max_bb3d_len
        if points_rotation.shape[0] < self.sample_num:
            offset = np.mean(points_rotation, 0) / max_bb3d_len
        else:
            offset = np.mean(points_rotation_sampled_normalized, 0)
        points_rotation_sampled_normalized = points_rotation_sampled_normalized - offset
        joint_rotation_normalized = joint_rotation_normalized - offset
        return points_rotation_sampled_normalized, joint_rotation_normalized, offset, coeff, max_bb3d_len

    def joint2pc(self, joint,radius=15):
        joint_num, _ = joint.shape

        radius = np.random.rand(joint_num, 100)*radius
        theta = np.random.rand(joint_num, 100)*np.pi
        phi = np.random.rand(joint_num, 100)*np.pi

        x = radius * np.sin(phi) * np.cos(theta)
        y = radius * np.sin(phi) * np.sin(theta)
        z = radius * np.cos(phi)

        point = np.tile(joint[:,np.newaxis,:], (1,100,1)) + np.concatenate((x[:,:,np.newaxis], y[:,:,np.newaxis], z[:,:,np.newaxis]), axis = -1)
        point = point.reshape([100*joint_num,3])
        sample = np.random.choice(100*joint_num, self.sample_num, replace=False)
        return point[sample, :]
    # pcl is point cloud N*3
    # pcl_img is depth image W*h*1

    def getpcl(self, imgD, com3D, cube, M):
        img_size = imgD.shape[0]
        mask = np.where(imgD > 0.99)
        dpt_ori = imgD * cube[2]/2.0 + com3D[2]
        # change the background value to 1
        dpt_ori[mask] = 0
        img_pcl = np.ones([img_size, img_size])*-1
        pcl = (self.depthToPCL(dpt_ori, M) - com3D)
        pcl_num = pcl.shape[0]
        cube_tile = np.tile(cube / 2.0, pcl_num).reshape([pcl_num, 3])
        pcl = pcl / cube_tile
        index_x = np.clip(np.floor((pcl[:, 0] + 1) / 2 * img_size), 0, img_size - 1).astype('int')
        index_y = np.clip(np.floor((1 - self.flip * pcl[:, 1]) / 2 * img_size), 0, img_size - 1).astype('int')
        img_pcl[index_y, index_x] = pcl[:, 2]
        return pcl, img_pcl


    def farthest_point_sample(self, xyz, npoint):
        N, C = xyz.shape
        S = npoint
        if N < S:
            centroids = np.arange(N)
            centroids = np.append(centroids, np.random.choice(centroids, size=S - N, replace=False))
        else:
            centroids = np.zeros(S).astype(np.int)
            distance = np.ones(N) * 1e10
            farthest = np.random.randint(0, S)
            for i in range(S):
                centroids[i] = farthest
                centroid = xyz[farthest, :]
                dist = np.sum((xyz - centroid) ** 2, -1)
                mask = dist < distance
                distance[mask] = dist[mask]
                farthest = distance.argmax()
        return np.unique(centroids)

    def depthToPCL(self, dpt, T, background_val=0.):
        fx, fy, fu, fv = self.paras
        # get valid points and transform
        pts = np.asarray(np.where(~np.isclose(dpt, background_val))).transpose()
        pts = np.concatenate([pts[:, [1, 0]] + 0.5, np.ones((pts.shape[0], 1), dtype='float32')], axis=1)
        pts = np.dot(np.linalg.inv(np.asarray(T)), pts.T).T
        pts = (pts[:, 0:2] / pts[:, 2][:, None]).reshape((pts.shape[0], 2))

        # replace the invalid data
        depth = dpt[(~np.isclose(dpt, background_val))]

        # get x and y data in a vectorized way
        row = (pts[:, 0] - fu) / fx * depth
        col = self.flip * (pts[:, 1] - fv) / fy * depth

        # combine x,y,depth
        return np.column_stack((row, col, depth))

    #tensor
    def unnormal_joint_img(self, joint_img):
        device = joint_img.device
        joint = torch.zeros(joint_img.size()).to(device)
        joint[:, :, 0:2] = (joint_img[:, :, 0:2] + 1)/2 * self.img_size
        joint[:, :, 2] = (joint_img[:, :, 2] + 1)/2 * self.cube_size[2]
        return joint

    def jointsImgTo3D(self, joint_uvd):
        joint_xyz = torch.zeros_like(joint_uvd)
        joint_xyz[:, :, 0] = (joint_uvd[:, :, 0]-self.paras[2])*joint_uvd[:, :, 2]/self.paras[0]
        joint_xyz[:, :, 1] = self.flip * (joint_uvd[:, :, 1]-self.paras[3])*joint_uvd[:, :, 2]/self.paras[1]
        joint_xyz[:, :, 2] = joint_uvd[:, :, 2]
        return joint_xyz

    def joints3DToImg(self, joint_xyz):
        fx, fy, fu, fv = self.paras
        joint_uvd = torch.zeros_like(joint_xyz)
        joint_uvd[:, :, 0] = (joint_xyz[:, :, 0] * fx / joint_xyz[:,:, 2] + fu)
        joint_uvd[:, :, 1] = (self.flip * joint_xyz[:, :, 1] * fy / joint_xyz[:, :, 2] + fv)
        joint_uvd[:, :, 2] = joint_xyz[:, :, 2]
        return joint_uvd

    def uvd_nl2xyznl_tensor(self, joint_uvd, M, cube_size, center):
        batch_size, joint_num, _ = joint_uvd.size()
        device = joint_uvd.device
        joint_img = torch.zeros_like(joint_uvd)
        cube_size_t = cube_size.to(device).view(batch_size, 1, 3).repeat(1, joint_num, 1)
        center_t = center.to(device).view(batch_size, 1, 3).repeat(1, joint_num, 1)
        M_t = M.to(device).view(batch_size, 1, 3, 3).repeat(1, joint_num, 1, 1)
        M_inverse = torch.inverse(M_t)
        joint_img[:, :, 0:2] = (joint_uvd[:, :, 0:2] + 1) * (self.img_size/2)
        joint_img[:, :, 2] = (joint_uvd[:, :, 2]) * (cube_size_t[:, :, 2] / 2.0) + center_t[:, :, 2]
        joint_uvd = self.get_trans_points(joint_img, M_inverse)
        joint_xyz = self.jointsImgTo3D(joint_uvd)
        joint_xyz = (joint_xyz - center_t) / (cube_size_t / 2.0)
        return joint_xyz

    def xyz_nl2uvdnl_tensor(self, joint_xyz, M, cube_size, center):
        device = joint_xyz.device
        batch_size, joint_num, _ = joint_xyz.size()
        joint_normal = torch.zeros_like(joint_xyz)
        cube_size_t = cube_size.to(device).view(batch_size, 1, 3).repeat(1, joint_num, 1)
        center_t = center.to(device).view(batch_size, 1, 3).repeat(1, joint_num, 1)
        M_t = M.to(device).view(batch_size, 1, 3, 3).repeat(1, joint_num, 1, 1)

        joint_temp = joint_xyz * cube_size_t / 2.0 + center_t
        joint_uvd = self.joints3DToImg(joint_temp)
        joint_uvd = self.get_trans_points(joint_uvd, M_t)
        joint_normal[:,:,0:2] = joint_uvd[:,:,0:2]/self.img_size*2 - 1
        joint_normal[:,:,2] = (joint_uvd[:,:,2] - center_t[:,:,2]) / (cube_size_t[:,:,2]/2)
        return joint_normal

    def get_trans_points(self, joints, M):
        device = joints.device
        joints_trans = torch.zeros_like(joints)
        joints_mat = torch.cat((joints[:, :, 0:2], torch.ones(joints.size(0), joints.size(1), 1).to(device)), dim=-1)
        joints_trans[:, :, 0:2] = torch.matmul(M, joints_mat.unsqueeze(-1)).squeeze(-1)[:, :, 0:2]
        joints_trans[:, :, 2] = joints[:, :, 2]
        return joints_trans


class nyu_loader(loader):
    def __init__(self, root_dir, img_type, aug_para = [10, 0.1, 180], img_size=128, cube_size=[300,300,300], center_type='refine', joint_num=23, loader=nyu_reader):
        super(nyu_loader, self).__init__(root_dir, img_type, img_size,center_type, 'nyu')
        self.paras = (588.03, 587.07, 320., 240.)
        self.cube_size = np.array(cube_size)
        self.allJoints = True
        self.flip = -1

        self.croppedSz = img_size
        self.joint_num = joint_num
        self.allJoints = True
        self.aug_modes = ['rot','com','sc','none']
        self.aug_para = aug_para
        # self.aug_modes = ['none']

        data_path = '{}/{}'.format(self.root_dir, self.img_type)
        label_path = '{}/joint_data.mat'.format(data_path)
        center_path ='{}/center_{}_refined.txt'.format(data_path, self.img_type)
        print('loading data...')
        self.labels = sio.loadmat(label_path)
        self.data_path = data_path
        self.all_joints_uvd = self.labels['joint_uvd'][0][:, joint_select, :][:, calculate, :]
        self.all_joints_xyz = self.labels['joint_xyz'][0][:, joint_select, :][:, calculate, :]
        self.refine_center_xyz = np.loadtxt(center_path)
        print('finish!!')
        if center_type =='refine':
            self.center_xyz = self.refine_center_xyz
        elif center_type =='joint':
            self.center_xyz = self.all_joints_xyz[:,20,:]
        elif center_type =='joint_mean':
            self.center_xyz = self.all_joints_xyz.mean(1)
        elif center_type == 'random':
            self.center_xyz = self.all_joints_xyz.mean(1)

        self.loader = loader
        self.test_cubesize = np.ones([8252, 3])*self.cube_size
        self.test_cubesize[2440:, :] = self.test_cubesize[2440:, :] * 5.0 / 6.0
        if joint_num == 14:
            self.allJoints = False
        else:
            self.allJoints = True

    def __getitem__(self, index):
        # timer = time.time()
        img_path = self.data_path + '/depth_1_{:07d}.png'.format(index+1)
        depth = self.loader(img_path)
        joint_uvd = self.all_joints_uvd[index].copy()
        joint_xyz = self.all_joints_xyz[index].copy()

        if self.img_type == 'test':
            cube_size = self.test_cubesize[index]
        else:
            cube_size = self.cube_size

        if self.center_type == 'mass':
            center_uvd = get_center_adopt(depth)
            center_xyz = self.jointImgTo3D(center_uvd)
        elif self.center_type == 'random':
            random_trans = (np.random.rand(3) - 0.5) * 2 * 0.2 * cube_size
            center_xyz = self.center_xyz[index] + random_trans
            center_uvd = self.joint3DToImg(center_xyz)
        else:
            center_xyz = self.center_xyz[index]
            center_uvd = self.joint3DToImg(center_xyz)

        gt3Dcrop = joint_xyz - center_xyz
        depth_crop, trans = self.Crop_Image_deep_pp(depth, center_uvd, cube_size, dsize=(self.img_size,self.img_size))

        if self.img_type == 'train':
            mode, off, rot, sc = self.rand_augment(sigma_com=self.aug_para[0], sigma_sc=self.aug_para[1], rot_range=self.aug_para[2])
            imgD, _, curLabel, curCube, com2D, M, _ = self.augmentCrop(depth_crop, gt3Dcrop, center_uvd, self.cube_size, trans, mode, off, rot, sc)
            curLabel = curLabel / (curCube[2] / 2.0)
        else:
            imgD = self.normalize_img(depth_crop.max(),depth_crop, center_xyz, cube_size)
            curLabel = gt3Dcrop / (cube_size[2] / 2.0)
            curCube = np.array(cube_size)
            com2D = center_uvd
            M = trans

        com3D = self.jointImgTo3D(com2D)
        joint_img = transformPoints2D(self.joint3DToImg(curLabel * (curCube[0] / 2.0) + com3D), M)
        joint_img[:, 0:2] = joint_img[:, 0:2] / (self.img_size/2) - 1
        joint_img[:, 2] = (joint_img[:, 2] - com3D[2]) / (curCube[0] / 2.0)

        pcl, pcl_img = self.getpcl(imgD, center_xyz, curCube, M)
        pcl_index = np.arange(pcl.shape[0])
        pcl_num = pcl.shape[0]
        if pcl_num == 0:
            pcl_sample = np.zeros([self.sample_num, 3])
        else:
            if pcl_num < self.sample_num:
                tmp = math.floor(self.sample_num / pcl_num)
                index_temp = pcl_index.repeat(tmp)
                pcl_index = np.append(index_temp, np.random.choice(pcl_index, size=divmod(self.sample_num, pcl_num)[1], replace=False))
            select = np.random.choice(pcl_index, self.sample_num, replace=False)
            pcl_sample = pcl[select, :]

        data = torch.from_numpy(imgD).float()
        data = data.unsqueeze(0)
        # for point net
        pcl_normal,joint_normal,offset,coeff,max_bbx_len = self.pca_point(pcl*curCube[0]/2 + com3D, curLabel*curCube[0]/2 + com3D)
        pcl_normal = torch.from_numpy(pcl_normal.transpose(1, 0)).float()
        joint_normal = torch.from_numpy(joint_normal).float()
        offset = torch.from_numpy(offset).float()
        coeff = torch.from_numpy(coeff).float()
        max_bbx_len = torch.ones([1]).float()*max_bbx_len

        pcl_sample = torch.from_numpy(pcl_sample.transpose(1,0)).float()
        joint_img = torch.from_numpy(joint_img).float()
        joint = torch.from_numpy(curLabel).float()
        center = torch.from_numpy(com3D).float()
        M = torch.from_numpy(M).float()
        cube = torch.from_numpy(curCube).float()

        # return data, pcl_sample, joint, joint_img, center, M, cube, pcl_normal, joint_normal, offset, coeff, max_bbx_len
        return data, joint, joint_img, center, M, cube,   

    def __len__(self):
        return len(self.all_joints_xyz)


class icvl_loader(loader):
    def __init__(self, root_dir, img_type, aug_para = [10, 0.1, 180], img_size=128, joint_num = 16, center_type='refine', loader=icvl_reader, full_img = False):
        super(icvl_loader, self).__init__(root_dir, img_type, img_size,center_type, 'icvl')
        self.paras = (240.99, 240.96, 160.0, 120.0)
        self.cube_size = [250, 250, 250]
        self.flip = 1
        self.joint_num = joint_num
        self.full_img = full_img

        self.img_size = img_size
        self.data_path = self.root_dir
        self.loader = loader

        self.all_joints_xyz, self.all_joints_uvd, self.all_centers_xyz,self.all_centers_uvd,self.img_dirs = self.read_joints(self.root_dir,self.img_type)
        self.length = len(self.all_joints_xyz)
        self.allJoints = False
        self.aug_modes = ['rot','com','sc','none']#,'com','sc','none'
        self.aug_para = aug_para
        if center_type =='refine':
            self.centers_xyz = self.all_centers_xyz
        else:
            self.centers_xyz = self.all_joints_xyz[:, 0, :]

    def __getitem__(self, index):
        img_path = self.img_dirs[index]
        if not os.path.exists(img_path):
            index = index + 1
            img_path = self.img_dirs[index]
        depth = self.loader(img_path)

        joint_uvd = self.all_joints_uvd[index].copy()
        # joint_xyz = self.all_joints_xyz[index].copy()
        joint_xyz = np.zeros_like(joint_uvd)
        joint_xyz[:, 0], joint_xyz[:, 1] = pixel2world_noflip(joint_uvd[:,0],joint_uvd[:,1],joint_uvd[:,2],self.paras)
        joint_xyz[:, 2] = joint_uvd[:, 2]


        center_xyz = self.centers_xyz[index].copy()
        center_uvd = self.joint3DToImg(center_xyz)
        gt3Dcrop = joint_xyz - center_xyz

        depth_crop, trans = self.Crop_Image_deep_pp(depth, center_uvd, self.cube_size,(self.img_size,self.img_size))

        if self.img_type == 'train':
            mode, off, rot, sc = self.rand_augment(sigma_com=self.aug_para[0], sigma_sc=self.aug_para[1], rot_range=self.aug_para[2])
            imgD, _, curLabel, cube, com2D, M, _ = self.augmentCrop(depth_crop, gt3Dcrop, center_uvd, self.cube_size, trans, mode, off, rot, sc)
            curLabel = curLabel / (cube[2] / 2.0)
        else:
            imgD = self.normalize_img(depth_crop.max(),depth_crop, center_xyz, self.cube_size)
            curLabel = gt3Dcrop / (self.cube_size[2] / 2.0)
            cube = np.array(self.cube_size)
            com2D = center_uvd
            M = trans

        com3D = self.jointImgTo3D(com2D)
        joint_img = transformPoints2D(self.joint3DToImg(curLabel * (cube[0] / 2.0) + com3D), M)
        joint_img[:, 0:2] = joint_img[:, 0:2] / (self.img_size/2) - 1
        joint_img[:, 2] = (joint_img[:, 2] - com3D[2]) / (cube[0] / 2.0)

        pcl, pcl_img = self.getpcl(imgD, center_xyz, cube, M)
        pcl_index = np.arange(pcl.shape[0])
        pcl_num = pcl.shape[0]
        if pcl_num == 0:
            pcl_sample = np.zeros([self.sample_num, 3])
        else:
            if pcl_num < self.sample_num:
                tmp = math.floor(self.sample_num / pcl_num)
                index_temp = pcl_index.repeat(tmp)
                pcl_index = np.append(index_temp, np.random.choice(pcl_index, size=divmod(self.sample_num, pcl_num)[1], replace=False))
            select = np.random.choice(pcl_index, self.sample_num, replace=False)
            pcl_sample = pcl[select, :]

        data = torch.from_numpy(imgD).float()
        data = data.unsqueeze(0)
        # for point net
        pcl_normal,joint_normal,offset,coeff,max_bbx_len = self.pca_point(pcl*cube[0]/2 + com3D, curLabel*cube[0]/2 + com3D)
        pcl_normal = torch.from_numpy(pcl_normal.transpose(1, 0)).float()
        joint_normal = torch.from_numpy(joint_normal).float()
        offset = torch.from_numpy(offset).float()
        coeff = torch.from_numpy(coeff).float()
        max_bbx_len = torch.ones([1]).float()*max_bbx_len


        pcl_sample = torch.from_numpy(pcl_sample.transpose(1,0)).float()
        joint_img = torch.from_numpy(joint_img).float()
        joint = torch.from_numpy(curLabel).float()
        center = torch.from_numpy(com3D).float()
        M = torch.from_numpy(M).float()
        cube = torch.from_numpy(cube).float()

        return data, pcl_sample, joint, joint_img, center, M, cube, pcl_normal, joint_normal, offset, coeff, max_bbx_len

    def read_joints(self, data_rt,img_type):
        if img_type =='train':
            f = open(data_rt + "/train.txt", "r")
            f_center = open(data_rt+"/center_train_refined.txt")
            lines = f.readlines()
            lines_center = f_center.readlines()
            f.close()
        else:
            f1 = open(data_rt+"/test_seq_1.txt", "r")
            f2 = open(data_rt + "/test_seq_2.txt", "r")
            f_center = open(data_rt + "/center_test_refined.txt")
            lines = f1.readlines()+f2.readlines()
            lines_center = f_center.readlines()
            f1.close()
            f2.close()

        centers_xyz = []
        centers_uvd = []
        joints_xyz = []
        joints_uvd = []

        img_names = []
        subSeq = ['0']
        for index, line in enumerate(lines):
            strs = line.split()
            p = strs[0].split('/')
            if not self.full_img:
                if ('0' in subSeq) and len(p[0]) > 6:
                    pass
                elif not ('0' in subSeq) and len(p[0]) > 6:
                    continue
                elif (p[0] in subSeq) and len(p[0]) <= 6:
                    pass
                elif not (p[0] in subSeq) and len(p[0]) <= 6:
                    continue

            img_path = data_rt + '/Depth/' + strs[0]
            if not os.path.isfile(img_path):
                continue

            joint_uvd = np.array(map(float, strs[1:])).reshape(16, 3)
            strs_center = lines_center[index].split()

            if strs_center[0] == 'invalid':
                continue
            else:
                center_xyz = np.array(map(float, strs_center)).reshape(3)
            center_uvd = np.zeros_like(center_xyz)
            center_uvd[0], center_uvd[1] = world2pixel(center_xyz[0], center_xyz[1], center_xyz[2], self.paras)
            center_uvd[2] = center_xyz[2]

            centers_xyz.append(center_xyz)
            centers_uvd.append(center_uvd)

            joint_xyz = np.zeros_like(joint_uvd)
            joint_xyz[:, 0], joint_xyz[:, 1] = pixel2world(joint_uvd[:, 0], joint_uvd[:, 1], joint_uvd[:, 2], self.paras)
            joint_xyz[:, 2] = joint_uvd[:, 2]
            joints_xyz.append(joint_xyz)
            joints_uvd.append(joint_uvd)
            img_names.append(img_path)

        f_center.close()
        return joints_xyz, joints_uvd, centers_xyz, centers_uvd, img_names

    def __len__(self):
        return self.length


class hands17_loader(loader):
    def __init__(self, root_dir, img_type, aug_para = [10, 0.1, 180], img_size=128, cube_size=[200,200,200], joint_num=21, img_num=957032, loader=hands17_reader):
        super(hands17_loader, self).__init__(root_dir, img_type, img_size, '', 'HANDS17')
        self.paras = (475.065948, 475.065857, 315.944855, 245.287079)
        self.aug_para = aug_para
        self.cube_size = cube_size
        self.flip = 1
        self.joint_num = joint_num
        self.img_num = img_num

        self.img_size = img_size
        self.data_path = self.root_dir
        self.loader = loader

        print('loading data...')
        self.all_joints_xyz, self.all_joints_uvd, self.all_centers_xyz, self.all_centers_uvd,self.img_dirs = self.read_joints(self.root_dir,self.img_type)
        print('finish!!')
        self.length = len(self.all_joints_xyz)
        self.allJoints = False
        self.aug_modes = ['com','sc','none','rot']#,'com','sc','none''rot','com',
        print('aug_mode',self.aug_modes)

    def __getitem__(self, index):
        img_path = self.img_dirs[index]
        if not os.path.exists(img_path):
            index = index + 1
            img_path = self.img_dirs[index]
        depth = self.loader(img_path)

        joint_uvd = self.all_joints_uvd[index].copy()
        joint_xyz = self.all_joints_xyz[index].copy()

        center_uvd = self.all_centers_uvd[index].copy()
        center_xyz = self.all_centers_xyz[index].copy()
        gt3Dcrop = joint_xyz - center_xyz

        depth_crop, trans = self.Crop_Image_deep_pp(depth, center_uvd, self.cube_size,(self.img_size,self.img_size))

        if self.img_type == 'train':
            mode, off, rot, sc = self.rand_augment(sigma_com=self.aug_para[0], sigma_sc=self.aug_para[1], rot_range=self.aug_para[2])#10, 0.1, 180
            imgD, _, curLabel, cube, com2D, M, _ = self.augmentCrop(depth_crop, gt3Dcrop, center_uvd, self.cube_size, trans, mode, off, rot, sc)
            curLabel = curLabel / (cube[2] / 2.0)
        else:
            imgD = self.normalize_img(depth_crop.max(), depth_crop, center_xyz, self.cube_size)
            curLabel = gt3Dcrop / (self.cube_size[2] / 2.0)
            cube = np.array(self.cube_size)
            com2D = center_uvd
            M = trans

        com3D = self.jointImgTo3D(com2D)
        joint_img = transformPoints2D(self.joint3DToImg(curLabel * (cube[0] / 2.0) + com3D), M)
        joint_img[:, 0:2] = joint_img[:, 0:2] / (self.img_size/2) - 1
        joint_img[:, 2] = (joint_img[:, 2] - com3D[2]) / (cube[0] / 2.0)

        pcl, pcl_img = self.getpcl(imgD, center_xyz, cube, M)
        pcl_index = np.arange(pcl.shape[0])
        pcl_num = pcl.shape[0]
        if pcl_num == 0:
            pcl_sample = np.zeros([self.sample_num, 3])
        else:
            if pcl_num < self.sample_num:
                tmp = math.floor(self.sample_num / pcl_num)
                index_temp = pcl_index.repeat(tmp)
                pcl_index = np.append(index_temp, np.random.choice(pcl_index, size=divmod(self.sample_num, pcl_num)[1], replace=False))
            select = np.random.choice(pcl_index, self.sample_num, replace=False)
            pcl_sample = pcl[select, :]

        data = torch.from_numpy(imgD).float()
        data = data.unsqueeze(0)

        # for point net
        pcl_normal,joint_normal,offset,coeff,max_bbx_len = self.pca_point(pcl*cube[0]/2 + com3D, curLabel*cube[0]/2 + com3D)
        pcl_normal = torch.from_numpy(pcl_normal.transpose(1, 0)).float()
        joint_normal = torch.from_numpy(joint_normal).float()
        offset = torch.from_numpy(offset).float()
        coeff = torch.from_numpy(coeff).float()
        max_bbx_len = torch.ones([1]).float()*max_bbx_len


        pcl_sample = torch.from_numpy(pcl_sample.transpose(1,0)).float()
        joint_img = torch.from_numpy(joint_img).float()
        joint = torch.from_numpy(curLabel).float()
        center = torch.from_numpy(com3D).float()
        M = torch.from_numpy(M).float()
        cube = torch.from_numpy(cube).float()

        return data, pcl_sample, joint, joint_img, center, M, cube, pcl_normal, joint_normal, offset, coeff, max_bbx_len

    def read_joints(self, data_rt,img_type):
        centers_xyz = []
        centers_uvd = []
        joints_xyz = []
        joints_uvd = []
        img_names = []
        if img_type =='train':
            f = open(data_rt + "/training/Training_Annotation.txt", "r")
            f_center = open(data_rt+"/center_train_refined.txt")
            lines = f.readlines()
            lines_center = f_center.readlines()
            f.close()
            f_center.close()

            for index, line in enumerate(lines):
                if index > self.img_num:
                    break
                strs = line.split()
                img_path = data_rt + '/training/images/' + strs[0]
                if not os.path.isfile(img_path):
                    continue

                joint_xyz = np.array(map(float, strs[1:])).reshape(self.joint_num, 3)
                strs_center = lines_center[index].split()

                if strs_center[0] == 'invalid':
                    continue
                else:
                    center_xyz = np.array(map(float, strs_center)).reshape(3)
                center_uvd = self.joint3DToImg(center_xyz)
                centers_xyz.append(center_xyz)
                centers_uvd.append(center_uvd)

                joint_uvd = self.joint3DToImg(joint_xyz)
                joints_xyz.append(joint_xyz)
                joints_uvd.append(joint_uvd)
                img_names.append(img_path)
        else:
            f = open(data_rt+"/frame/BoundingBox.txt", "r")
            f_center = open(data_rt + "/center_test_refined.txt")
            lines = f.readlines()
            lines_center = f_center.readlines()
            f.close()
            f_center.close()
            for index, line in enumerate(lines):
                if index > self.img_num:
                    break
                strs = line.split()
                img_path = data_rt + '/frame/images/' + strs[0]
                if not os.path.isfile(img_path):
                    continue
                strs_center = lines_center[index].split()
                if strs_center[0] == 'invalid':
                    continue
                else:
                    center_xyz = np.array(map(float, strs_center)).reshape(3)
                center_uvd = self.joint3DToImg(center_xyz)
                centers_xyz.append(center_xyz)
                centers_uvd.append(center_uvd)

                joints_xyz.append(np.ones([self.joint_num, 3]))
                joints_uvd.append(np.ones([self.joint_num, 3]))
                img_names.append(img_path)

        return joints_xyz, joints_uvd, centers_xyz, centers_uvd, img_names


    def __len__(self):
        return self.length

    # def jointsImgTo3D(self, joint_uvd):
    #     joint_xyz = torch.zeros_like(joint_uvd)
    #     joint_xyz[:, :, 0] = (joint_uvd[:, :, 0]-self.paras[2])*joint_uvd[:, :, 2]/self.paras[0]
    #     joint_xyz[:, :, 1] = (joint_uvd[:, :, 1]-self.paras[3])*joint_uvd[:, :, 2]/self.paras[1]
    #     joint_xyz[:, :, 2] = joint_uvd[:, :, 2]
    #     return joint_xyz
    #
    # def uvd_nl2xyznl_tensor(self, joint_uvd, M, cube_size, center):
    #     batch_size, joint_num, _ = joint_uvd.size()
    #     device = joint_uvd.device
    #     joint_img = torch.zeros_like(joint_uvd)
    #     cube_size_t = cube_size.to(device).view(batch_size, 1, 3).repeat(1, joint_num, 1)
    #     center_t = center.to(device).view(batch_size, 1, 3).repeat(1, joint_num, 1)
    #     M_t = M.to(device).view(batch_size, 1, 3, 3).repeat(1, joint_num, 1, 1)
    #     M_inverse = torch.inverse(M_t)
    #     joint_img[:, :, 0:2] = (joint_uvd[:, :, 0:2] + 1) * (self.img_size/2)
    #     joint_img[:, :, 2] = (joint_uvd[:, :, 2]) * (cube_size_t[:, :, 2] / 2.0) + center_t[:, :, 2]
    #     joint_uvd = self.get_trans_points(joint_img, M_inverse)
    #     joint_xyz = self.jointsImgTo3D(joint_uvd)
    #     joint_xyz = (joint_xyz - center_t) / (cube_size_t / 2.0)
    #     return joint_xyz
    #
    # def get_trans_points(self, joints, M):
    #     device = joints.device
    #     joints_trans = torch.zeros_like(joints)
    #     joints_mat = torch.cat((joints[:, :, 0:2], torch.ones(joints.size(0), joints.size(1), 1).to(device)), dim=-1)
    #     joints_trans[:, :, 0:2] = torch.matmul(M, joints_mat.unsqueeze(-1)).squeeze(-1)[:, :, 0:2]
    #     joints_trans[:, :, 2] = joints[:, :, 2]
    #     return joints_trans


class msra_loader(loader):
        def __init__(self,  root_dir, img_type, aug_para = [10, 0.1, 180], img_size=128,  joint_num = 21, center_type='refine', test_persons = [], loader=msra_reader):
            super(msra_loader, self).__init__(root_dir, img_type, img_size,center_type, 'msra')
            self.paras = (241.42, 241.42, 160, 120)
            self.cube_size = [200,200,200,180,180,180,170,160,150]
            self.centers_type = center_type
            self.aug_para = aug_para
            person_list =[0,1,2,3,4,5,6,7,8]
            train_persons = list(set(person_list).difference(set(test_persons)))
            self.flip = -1
            if img_type =='train':
                self.all_joints_xyz, self.all_joints_uvd, self.keys = self.read_joints(root_dir,img_type, persons=train_persons)
                self.length = len(self.all_joints_xyz)
            else:
                self.all_joints_xyz, self.all_joints_uvd, self.keys = self.read_joints(root_dir,img_type, persons=test_persons)
                self.length = len(self.all_joints_xyz)
            file_uvd = open('./msra_label.txt', 'w')
            for index in range(len(self.all_joints_uvd)):
                np.savetxt(file_uvd, self.all_joints_uvd[index].reshape([1,joint_num * 3]), fmt='%.3f')
            if center_type == 'refine' or center_type =='mass':
                file_name = self.root_dir+'/center_'+img_type+'_'+str(test_persons[0])+'_refined.txt'
                self.centers_xyz = np.loadtxt(file_name)
            else:
                self.centers_xyz = self.all_joints_xyz[:,5,:]
            self.loader = loader
            self.joint_num = joint_num
            self.aug_modes =['rot','com','sc','none']

        def __getitem__(self, index):
            person = self.keys[index][0]
            name = self.keys[index][1]
            cube_size = [self.cube_size[person],self.cube_size[person],self.cube_size[person]]
            file = '%06d' % int(self.keys[index][2])

            depth, pcl = msra_reader(self.root_dir+"/P" + str(person) + "/" + str(name) + "/" + str(file) + "_depth.bin",self.paras)
            assert (depth.shape == (240, 320))

            joint_uvd = self.all_joints_uvd[index].copy()
            joint_xyz = np.zeros_like(joint_uvd)
            joint_xyz[:, 0], joint_xyz[:, 1] = pixel2world(joint_uvd[:, 0], joint_uvd[:, 1], joint_uvd[:, 2],self.paras)
            joint_xyz[:, 2] = joint_uvd[:, 2]

            if self.centers_type == 'mass':
                center_uvd = get_center_fast(depth)
                center_xyz = self.jointImgTo3D(center_uvd)
            else:
                center_xyz = self.centers_xyz[index].copy()
                center_uvd = self.joint3DToImg(center_xyz)
            gt3Dcrop = joint_xyz - center_xyz

            depth_crop, trans = self.Crop_Image_deep_pp(depth, center_uvd, cube_size, dsize=(self.img_size, self.img_size))

            if self.img_type == 'train':
                mode, off, rot, sc = self.rand_augment(sigma_com=self.aug_para[0], sigma_sc=self.aug_para[1], rot_range=self.aug_para[2])
                imgD, _, curLabel, curCube, com2D, M, _ = self.augmentCrop(depth_crop, gt3Dcrop, center_uvd,cube_size, trans, mode, off, rot, sc)
                curLabel = curLabel / (curCube[2] / 2.0)
            else:
                imgD = self.normalize_img(depth_crop.max(), depth_crop, center_xyz, cube_size)
                curLabel = gt3Dcrop / (cube_size[2] / 2.0)
                curCube = np.array(cube_size)
                com2D = center_uvd
                M = trans

            com3D = self.jointImgTo3D(com2D)
            joint_img = transformPoints2D(self.joint3DToImg(curLabel * (curCube[0] / 2.0) + com3D), M)
            joint_img[:, 0:2] = joint_img[:, 0:2] / (self.img_size / 2) - 1
            joint_img[:, 2] = (joint_img[:, 2] - com3D[2]) / (curCube[0] / 2.0)

            pcl, pcl_img = self.getpcl(imgD, center_xyz, curCube, M)
            pcl_index = np.arange(pcl.shape[0])
            pcl_num = pcl.shape[0]
            if pcl_num == 0:
                pcl_sample = np.zeros([self.sample_num, 3])
            else:
                if pcl_num < self.sample_num:
                    tmp = math.floor(self.sample_num / pcl_num)
                    index_temp = pcl_index.repeat(tmp)
                    pcl_index = np.append(index_temp,
                                          np.random.choice(pcl_index, size=divmod(self.sample_num, pcl_num)[1],
                                                           replace=False))
                select = np.random.choice(pcl_index, self.sample_num, replace=False)
                pcl_sample = pcl[select, :]

            data = torch.from_numpy(imgD).float()
            data = data.unsqueeze(0)
            # for point net
            pcl_normal, joint_normal, offset, coeff, max_bbx_len = self.pca_point(pcl * curCube[0] / 2 + com3D,
                                                                                  curLabel * curCube[0] / 2 + com3D)
            pcl_normal = torch.from_numpy(pcl_normal.transpose(1, 0)).float()
            joint_normal = torch.from_numpy(joint_normal).float()
            offset = torch.from_numpy(offset).float()
            coeff = torch.from_numpy(coeff).float()
            max_bbx_len = torch.ones([1]).float() * max_bbx_len

            pcl_sample = torch.from_numpy(pcl_sample.transpose(1, 0)).float()
            joint_img = torch.from_numpy(joint_img).float()
            joint = torch.from_numpy(curLabel).float()
            center = torch.from_numpy(com3D).float()
            M = torch.from_numpy(M).float()
            cube = torch.from_numpy(curCube).float()

            return data, pcl_sample, joint, joint_img, center, M, cube, pcl_normal, joint_normal, offset, coeff, max_bbx_len

        # return joint_uvd
        def read_joints(self, data_rt, img_type, persons=[0, 1, 2, 3, 4, 5, 6, 7],poses=["1", "2", "3", "4", '5', '6', '7', '8', '9', 'I', 'IP', 'L', 'MP', 'RP', 'T', 'TIP','Y']):
            joints_xyz = []
            joints_uvd = []
            index = 0
            keys = {}
            file_record = open('./msra_record_list.txt', "w")
            for person in persons:
                for pose in poses:
                    with open(data_rt + "/P" + str(person) + "/" + str(pose) + "/joint.txt") as f:
                        num_joints = int(f.readline())
                        for i in range(num_joints):
                            file_record.write('P'+ str(person) + "/" + str(pose) +'/'+'%06d' % int(i) + "_depth.bin"+'\r\n')
                            joint = np.fromstring(f.readline(), sep=' ')
                            joint_xyz = joint.reshape(21, 3)
                            joint_xyz[:,2] = -joint_xyz[:,2]
                            joint_uvd = np.zeros_like(joint_xyz)
                            #need to chaneg z to -
                            joint_uvd[:, 0], joint_uvd[:, 1] = world2pixel(joint_xyz[:, 0], joint_xyz[:, 1], joint_xyz[:, 2],self.paras)
                            joint_uvd[:, 2] = joint_xyz[:, 2]
                            # joint = joint.reshape(63)
                            joints_xyz.append(joint_xyz)
                            joints_uvd.append(joint_uvd)
                            keys[index] = [person, pose, i]
                            index += 1
            file_record.close()
            return joints_xyz, joints_uvd, keys

        def __len__(self):
            return self.length


class itop_loader(loader):
    def __init__(self, root_dir, img_type, dir, joint_num= 15, loader=icvl_reader):
        super(itop_loader, self).__init__(root_dir, img_type, img_size, 'itop')
        self.data_path = root_dir
        self.loader = loader
        self.joint_num = joint_num
        self.paras = (285.71, 285.71, 160.0, 120.0)# 1/0.0035
        self.cube_size = np.array([1.2, 2.0, 0.8]) #[2.0, 2.0, 2.0]

        self.all_joints_xyz, self.all_joints_uvd, self.all_centers_xyz, self.all_centers_uvd, self.depths = self.read_joints(root_dir,img_type,dir)
        self.length = len(self.all_joints_xyz)
        self.allJoints = False
        self.aug_modes = ['none','rot','com','sc'] #'none','rot','com','sc',

    def __getitem__(self, index):

        depth = self.depths[index].copy()

        joint_uvd = self.all_joints_uvd[index].copy()
        joint_xyz = self.all_joints_xyz[index].copy()

        center_uvd = self.all_centers_uvd[index].copy()
        center_xyz = self.all_centers_xyz[index].copy()
        gt3Dcrop = joint_xyz - center_xyz

        # use opencv to crop image
        # depth_crop, trans = Crop_Image_ren(depth, center_uvd, self.cube_size[0], self.para, self.img_size)
        # trans = np.concatenate((trans, np.array([[0, 0, 1]])), axis=0)

        depth_crop, trans = self.Crop_Image_deep_pp(depth, center_uvd, self.cube_size,(self.img_size,self.img_size))

        if self.img_type == 'train':
            mode, off, rot, sc = self.rand_augment(sigma_com=self.cube_size[0]*0.02, sigma_sc=0.1, rot_range=180)
            imgD, _, curLabel, cube, com2D, M, _ = self.augmentCrop(depth_crop, gt3Dcrop, center_uvd, self.cube_size, trans,
                                                                mode, off, rot, sc)
        else:
            imgD = normalize_img(depth_crop.max(), depth_crop, center_xyz, self.cube_size)
            cube = self.cube_size
            com2D = center_uvd
            M = trans
            curLabel = gt3Dcrop

        pcl, pcl_img = self.getpcl(imgD, center_xyz, cube, M)
        pcl_index = np.arange(pcl.shape[0])
        pcl_num = pcl.shape[0]
        if pcl_num == 0:
            pcl_sample = np.zeros([self.sample_num, 3])
        else:
            if pcl_num < self.sample_num:
                tmp = math.floor(self.sample_num / pcl_num)
                index_temp = pcl_index.repeat(tmp)
                pcl_index = np.append(index_temp, np.random.choice(pcl_index, size=divmod(self.sample_num, pcl_num)[1], replace=False))
            select = np.random.choice(pcl_index, self.sample_num, replace=False)
            pcl_sample = pcl[select, :]

        # for point net
        pcl_normal,joint_normal,offset,coeff,max_bbx_len = self.pca_point(pcl*cube[0]/2 + com3D, curLabel*cube[0]/2 + com3D)
        pcl_normal = torch.from_numpy(pcl_normal.transpose(1, 0)).float()
        joint_normal = torch.from_numpy(joint_normal).float()
        offset = torch.from_numpy(offset).float()
        coeff = torch.from_numpy(coeff).float()
        max_bbx_len = torch.ones([1]).float()*max_bbx_len


        cube_size_jn = np.tile((cube / 2.0), self.joint_num).reshape([self.joint_num, 3])
        curLabel = curLabel / cube_size_jn
        com3D = self.jointImgTo3D(com2D)
        joint_img = self.transformPoints2D(joint3DToImg(curLabel * cube_size_jn + com3D), M)
        joint_img[:, 0:2] = joint_img[:, 0:2] / (self.img_size/2) - 1
        joint_img[:, 2] = (joint_img[:, 2] - com3D[2]) / (cube[2] / 2.0)

        img = torch.from_numpy(imgD).float()
        img = img.unsqueeze(0)

        # edge_data = self.get_edges(imgD).float()
        # edge_data = edge_data.unsqueeze(0)

        joint_img = torch.from_numpy(joint_img).float()
        pcl_sample = torch.from_numpy(pcl_sample).float()
        pcl_img = torch.from_numpy(pcl_img).float().unsqueeze(0)
        joint = torch.from_numpy(curLabel).float()
        center = torch.from_numpy(com3D).float()
        M = torch.from_numpy(M).float()
        cube = torch.from_numpy(cube).float()
        return img, pcl_img, pcl_sample, torch.ones([1]), joint, joint_img, center, M, cube
        # return data, joint_xyz_crop, heatmap_xyz, center_xyz

    def read_joints(self, data_rt, img_type, dir):
        labels_dir = data_rt + '/ITOP_{}_{}_labels.h5'.format(dir, img_type)
        datas_dir = data_rt + '/ITOP_{}_{}_depth_map.h5'.format(dir, img_type)
        centers_dir = data_rt + '/ITOP_{}_{}_centers.txt'.format(dir, img_type)

        h5_labels = h5py.File(labels_dir, 'r')
        h5_imgs = h5py.File(datas_dir, 'r')

        depths = h5_imgs['data'][:]
        valid = h5_labels['is_valid'][:]
        joints_xyz = h5_labels['real_world_coordinates'][:]
        joints_uvd = np.zeros_like(joints_xyz)
        joints_uvd[:, :, 0:2] = h5_labels['image_coordinates'][:]
        joints_uvd[:, :, 2] = joints_xyz[:, :, 2]
        lines_centers = open(centers_dir).readlines()

        depths = depths[np.where(valid)]
        depths = depths.astype(np.float32)
        # depths = depths[:, :, :, np.newaxis]
        joints_xyz = joints_xyz[np.where(valid)]
        joints_uvd = joints_uvd[np.where(valid)]
        # filp
        joints_xyz[:, :, 1] = -joints_xyz[:, :, 1]
        centers_xyz = []
        for index in range(valid.shape[0]):
            if valid[index]:
                joints_uvd
                strs_center = lines_centers[index].split()
                centers_xyz.append(np.array(map(float, strs_center)).reshape(3))

        centers_xyz = np.array(centers_xyz)
        centers_xyz[:,1] = -centers_xyz[:,1]
        centers_uvd = joint3DToImg(centers_xyz, self.para)
        return joints_xyz, joints_uvd, centers_xyz, centers_uvd, depths


    def __len__(self):
        return self.length

    def jointsImgTo3D(self, joint_uvd):
        joint_xyz = torch.zeros_like(joint_uvd)
        joint_xyz[:, :, 0] = (joint_uvd[:, :, 0]-self.para[2])*joint_uvd[:, :, 2]/self.para[0]
        joint_xyz[:, :, 1] = (joint_uvd[:, :, 1]-self.para[3])*joint_uvd[:, :, 2]/self.para[1]
        joint_xyz[:, :, 2] = joint_uvd[:, :, 2]
        return joint_xyz

    def uvd_nl2xyznl_tensor(self, joint_uvd, M, cube_size, center):
        batch_size, joint_num, _ = joint_uvd.size()
        device = joint_uvd.device
        joint_img = torch.zeros_like(joint_uvd)
        cube_size_t = cube_size.to(device).view(batch_size, 1, 3).repeat(1, joint_num, 1)
        center_t = center.to(device).view(batch_size, 1, 3).repeat(1, joint_num, 1)
        M_t = M.to(device).view(batch_size, 1, 3, 3).repeat(1, joint_num, 1, 1)
        M_inverse = torch.inverse(M_t)
        joint_img[:, :, 0:2] = (joint_uvd[:, :, 0:2] + 1) * (self.img_size/2)
        joint_img[:, :, 2] = (joint_uvd[:, :, 2]) * (cube_size_t[:, :, 2] / 2.0) + center_t[:, :, 2]
        joint_uvd = self.get_trans_points(joint_img, M_inverse)
        joint_xyz = self.jointsImgTo3D(joint_uvd)
        joint_xyz = (joint_xyz - center_t) / (cube_size_t / 2.0)
        return joint_xyz

    def get_trans_points(self, joints, M):
        device = joints.device
        joints_trans = torch.zeros_like(joints)
        joints_mat = torch.cat((joints[:, :, 0:2], torch.ones(joints.size(0), joints.size(1), 1).to(device)), dim=-1)
        joints_trans[:, :, 0:2] = torch.matmul(M, joints_mat.unsqueeze(-1)).squeeze(-1)[:, :, 0:2]
        joints_trans[:, :, 2] = joints[:, :, 2]
        return joints_trans


def xyz2error(output, joint, center, cube_size):
    with torch.no_grad():
        batchsize, joint_num, _ = output.shape
        center = np.tile(center.reshape(batchsize, 1, 3), [1, joint_num, 1])
        cube_size = np.tile(cube_size.reshape(batchsize, 1, 3), [1, joint_num, 1])

        output = output * cube_size / 2 + center
        joint_xyz_label = joint * cube_size / 2 + center

        temp = (output - joint_xyz_label) * (output - joint_xyz_label)
        error = np.mean(np.sqrt(np.sum(temp, 2)))

    return error


def RunMyImageFloder():
    cuda_id = 0
    torch.cuda.set_device(cuda_id)
    joint_num = 21
    batch_size = 128
    input_size = 128
    inter_size = 128
    # test_data = nyu_loader('/home/ljs/pfren/dataset/nyu', 'test', cube_size=[300,300,300], center_type='refine', aug_para=[20, 0.1, 180])
    # test_data = itop_loader('/data/users/pfren/data/dataset/body/itop', 'test', 'side')
    # test_data = icvl_loader('/data/users/ljs/pfren/dataset/icvl', 'test', full_img=False)
    # test_data = msra_loader('/data/users/ljs/pfren/dataset/msra', 'test', test_persons=[0],center_type='refine')
    test_data = hands17_loader('/data/users/ljs/pfren/dataset/hands17', 'train', cube_size=[200,200,200],img_num=1000, aug_para=[0, 0.0, 0])
    GFM_ = G.GFM()
    dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=1)
    f = open('cube.txt','w')
    joint_world_pca = torch.Tensor([])
    joint_img_pca = torch.Tensor([])
    # joint_predict = np.loadtxt('/data/users/ljs/pfren/pycharm/multi_net/model/nyu/offset_multi_net_resnet18_Gsgd0.3_step_batch32_epoch80_ips128_its128_cube300_centerType_refineL1Lossaug_10_0.1_180_000_0_30_0_uvd/joint_uvd.txt')
    # joint_predict = joint_predict.reshape([-1,14,3])
    # num = 0
    # pca = PCA(n_components=35)
    for index, data in enumerate(dataloader):
        img, img_ori, joint_world, joint_img, center, M, cube, pcl_normal, joint_normal, offset, coeff, max_bbx_len = data
        # debug_2d_heatmap(img, joint_img, index, GFM_)
        # debug_offset_heatmap(img, joint_img, index, GFM_, 0.4)
        debug_2d_img(img,joint_img,index,'hands17')
        # debug_2d_img(img,joint_img,index,'hands17',batch_size)
        # debug_point_heatmap('nyu_full', data, index, GFM_)
        # debug_point_feature(data,index,GFM_)
        # joint_world_pca = torch.cat((joint_world_pca,joint_world),dim=0)
        # joint_img_pca = torch.cat((joint_img_pca,joint_img),dim=0)
        # num = num + img.size(0)
        # if num>100000:
        #     break
        # uvd = test_data.xyz_nl2uvdnl_tensor(joint_world, M, cube, center)
        # print((uvd-joint_img).sum())
        # [:, calculate,:]
        # joint_uvd = test_data.get_trans_points(torch.from_numpy(joint_predict[index]).view(1,14,3).float(), M)
        # joint_uvd[:,:,0:2] = joint_uvd[:,:,0:2]/64 - 1
        # debug_2d_img(img, joint_uvd , index, 'icvl', batch_size, dir='icvl')
        print(index)
    # joint_img_pca = joint_img_pca.numpy().reshape(-1,joint_num*3)
    # pca.fit(joint_img_pca)
    # file_pca = open('hands17_uvd_para.txt', 'w')
    # np.savetxt(file_pca, pca.components_)
    # np.savetxt('hands17_uvd_mean.txt', pca.mean_)
    #
    # joint_world_pca = joint_world_pca.numpy().reshape(-1,14*3)
    # pca.fit(joint_world_pca)
    # file_pca = open('hands17_xyz_para.txt', 'w')
    # np.savetxt(file_pca, pca.components_)
    # np.savetxt('hands17_xyz_mean.txt', pca.mean_)
        # output = test_data.uvd_nl2xyznl_tensor(joint_img.view(-1, joint_num, 3), M, cube, center)
        # error = xyz2error(output.view(-1, joint_num, 3).cpu().numpy(), joint_world.numpy(), center.numpy(), cube.numpy())
        # print(error)
        # debug_2d_heatmap(img, joint_img, index, GFM_, dir_name='crop_')
        # print(index)
        # debug_point_heatmap('nyu_full', data, index, GFM_)

        # a = torch.eye(3)*0.8
        # a[2,2] = 1
        # a[1,2] = 0.1
        # a[0,2] = 0.2
        # b = a.unsqueeze(0).repeat(batch_size, 1, 1)
        # grid = F.affine_grid(b[:, 0:2, :], (batch_size,1,inter_size,inter_size))
        # imgD_affine = F.grid_sample(img, grid)
        # joint_affine = torch.zeros_like(joint_img)
        # joint_affine = joint_img # [-imgsize/2,imgsize/2]
        # # cat_tensor = torch.Tensor([0,0,1]).view(1,1,3).repeat(img.size(0),1,1)
        # trans_M = b.inverse().unsqueeze(1).repeat(1, joint_num, 1, 1)
        # joint_affine[:, :, 0:2] = (torch.matmul(trans_M, joint_affine.unsqueeze(-1)).squeeze()+1)[:, :, 0:2]*inter_size/2
        # img_draw = (imgD_affine.detach().cpu().numpy() + 1) / 2 * 255
        # img_show = draw_pose('nyu_full', cv2.cvtColor(img_draw[0, 0], cv2.COLOR_GRAY2RGB), joint_affine[0])
        # cv2.imwrite('./debug/'+str(index)+'.png', img_show)
        # print(index)
    #     img, pcl_sample, joint_world, joint_img, center, M, cube, pcl_normal, joint_normal, offset, coeff, max_bbx_len = data
    #     np.savetxt(f, max_bbx_len.numpy(), fmt='%.2f')
    #     print(index)
    # f.close()

        # print(index, error)
        # debug_point_heatmap('msra',data,index,GFM_)
        # debug_2d_heatmap(data,index,GFM_)
        # print(index)
        # joints_uvd = (joint_uvd_label.view(batch_size, -1, 3).detach().cpu().numpy() + 1) * 64
        # img_draw = (data.cpu().numpy() + 1) / 2 * 255
        # for img_index in range(joints_uvd.shape[0]):
        #     img_dir = 'debug/' + str(batch_size * index + img_index) + '.png'
        #     img_show = draw_pose('itop', cv2.cvtColor(img_draw[img_index, 0], cv2.COLOR_GRAY2RGB),joints_uvd[img_index])
        #     cv2.imwrite(img_dir, img_show)

        #
        # heatmap2d = GFM_.joint2heatmap2d(joint_uvd_label, isFlip=False)
        # depth = GFM_.depth2map(joint_uvd_label[:, :, 2])
        # feature = heatmap2d * (depth + 1) / 2
        # for i in range(data.size(0)):
        #     img = data.numpy()[i]
        #     heatmap = feature.numpy()[i]
        #     img_heatmap = ((img + 1) / 4 + heatmap.sum(0) / 2)*255.0
        #     img_name_1 = './debug/heatmap_joint_' + str(index) + '_' +str(i) + '.png'
        #     cv2.imwrite(img_name_1, np.transpose(img_heatmap,(1,2,0)))
        # print(index)
        # print('all get data :' ,time.time() - timer)
        # error = heatmap2error(heatmap.numpy(),joint[:,:,2].numpy(),joint.numpy(),center.numpy(),test_data)
        # print(error)
        # joint_xyz_label = test_data.all_joints_xyz[index]
        # # output = unnorm_joint(joint.cpu().numpy()[0], center.cpu().numpy()[0],  test_data.para, test_data.cube_size)
        # _,output = trans_joints(joint.view(1,16,3).numpy(), center.numpy(), test_data.para, test_data.cube_size)
        # temp = (output[0] - joint_xyz_label) * (output[0] - joint_xyz_label)
        # error = np.mean(np.sqrt(np.sum(temp, 1)))
        # print(index, error)
        # joint_xyz,joint_uvd = heatmap2joint(heatmap[0], joint_depth[0], center, test_data.para, test_data.cube_size)
        # joint_label = joint[0]
        # joint_label[:,0],joint_label[:,1] = pixel2world(joint_label[:,0],joint_label[:,1],joint_label[:,2],test_data.para)
        # temp = (joint_xyz - joint_label)*(joint_xyz - joint_label)
        # error = torch.mean(torch.sqrt(torch.sum(temp, 1)))
        # print(error)
        # if error>0.02:
        #     print(index,error)

        # temp_img = data.numpy()
        # temp_heatmap= heatmap.numpy()
        # for i in range(temp_img.shape[0]):
        #     img = np.zeros((croppedSz, croppedSz, 1))
        #     heatmap = np.zeros((heatmapSz, heatmapSz, 1))
        #     img[:, :, 0] = (temp_img[i] + 1) / 2 * 255
        #     heatmap[:, :, 0] += (np.sum(temp_heatmap[i], 0)) * 255.0
        #     heatmap = cv2.resize(heatmap,None,fx=2, fy=2, interpolation = cv2.INTER_AREA)
        #     img_heatmap = img + heatmap.reshape(croppedSz,croppedSz,1)
        #     img_name_1 = './debug/img_' + str(index)+'_'+str(i) + '.png'
        #     cv2.imwrite(img_name_1, img)
        #     img_name_1 = './debug/heatmap_' + str(index)+'_'+str(i) + '.png'
        #     cv2.imwrite(img_name_1, img_heatmap)


        #for point feature
        # rot = np.random.rand(batch_size, 3) * 0
        # pcl_sample_rot = GFM_.rotation_points(pcl_sample, rot)
        # joint_rot = GFM_.rotation_points(joint, rot)
        # joint_pc = GFM_.joint2pc(joint_rot)
        # add_pc = torch.cat((pcl_sample_rot, joint_pc), dim=1)
        # img_pcl = GFM_.pcl2img(add_pc, img_size).squeeze(1).unsqueeze(-1)*255
        # # sort_ps = select_keypc(add_pc)
        # for i in range(data.size(0)):
        #     img = img_pcl.numpy()[i]
        #     img[img > 0] = 255 / 2.0
        #     img_name_1 = './debug/pcl_' + str(index)+'_'+str(i) + '.png'
        #     cv2.imwrite(img_name_1, img)
        #
        # print(index)

if __name__ == "__main__":
    from nyu_loader import draw

    root = '/home/ljs/pfren/dataset/nyu'
    dataset = iter(nyu_loader(root, 'test'))
    for i in range(1, 10):
        data = next(dataset) # 'trans'
        img, j3d_xyz, j3d_uvd, center_xyz, M, cube = data
        # for item in data:
        #     print(item.size())
        # print(j3d_xyz)
        j3d_uvd = (j3d_uvd + 1) * 64
        
        
        a = draw(img[0].numpy(), j3d_uvd[:,:2].numpy())
        cv2.imwrite("./ori_%d_j2d.png" % i, (a+1)*100)

    print('done')


