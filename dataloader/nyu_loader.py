import cv2
import numpy as np
import scipy.io as sio
from glob import glob

from dataloader.loader import Loader
from util.util import uvd2xyz, xyz2uvd

JOINT = np.array([0,1,3,5,   6,7,9,11,  12,13,15,17,  18,19,21,23,  24,25,27,28,  32,30,31])
# select 14 joints for evaluation 
EVAL = np.array([0, 2, 4, 6, 8, 10, 12, 14, 16, 17, 18, 21, 22, 20])


class NYU(Loader):
    
    def __init__(self, root, phase, val=False, img_size=128, aug_para=[10, 0.1, 180], cube=[300,300,300], jt_num=14):
        super(NYU, self).__init__(root, phase, img_size, 'nyu')
        self.name = 'nyu'
        self.root = root
        self.phase = phase
        self.val = val

        self.paras = (588.03, 587.07, 320., 240.)
        self.cube= np.asarray(cube)
        self.dsize = np.asarray([img_size, img_size])
        self.img_size = img_size

        self.jt_num = jt_num
        self.aug_para = aug_para

        self.data = self.make_dataset()
        self.test_cube = np.ones([8252, 3]) * self.cube
        self.test_cube[2440:, :] = self.test_cube[2440:, :] * 5.0 / 6.0
        self.flip = -1 # flip y axis when doing xyz <-> uvd transformation

        print("loading dataset, containing %d images." % len(self.data))

    def __getitem__(self, index):
        img = self.nyu_reader(self.data[index][0])
        jt_xyz = self.data[index][2].copy()

        if self.phase == 'test':
            cube = self.test_cube[index]
        else:
            cube = self.cube

        center_xyz = self.data[index][3].copy()
        center_uvd = xyz2uvd(center_xyz, self.paras, self.flip)

        jt_xyz -= center_xyz
        img, M = self.crop(img, center_uvd, cube, self.dsize)

        if self.phase == 'train' and self.val == False:
            aug_op, trans, scale, rot = self.random_aug(*self.aug_para)
            img, jt_xyz, cube, center_uvd, M = self.augment(img, jt_xyz, center_uvd, cube, M, aug_op, trans, scale, rot)
            center_xyz = uvd2xyz(center_uvd, self.paras, self.flip)
        else:
            img = self.normalize(img.max(), img, center_xyz, cube)


        jt_uvd = self.transform_jt_uvd(xyz2uvd(jt_xyz + center_xyz, self.paras, self.flip), M)
        jt_uvd[:, :2] = jt_uvd[:, :2] / (self.img_size / 2.) - 1
        jt_uvd[:, 2] = (jt_uvd[:, 2] - center_xyz[2]) / (cube[2] / 2.0)
        jt_xyz = jt_xyz / (cube / 2.)

        return img[np.newaxis, :].astype(np.float32), jt_xyz.astype(np.float32), jt_uvd.astype(np.float32), center_xyz.astype(np.float32), M.astype(np.float32), cube.astype(np.float32)

    def __len__(self):
        return len(self.data)

    def nyu_reader(self, img_path):
        img = cv2.imread(img_path)
        depth = np.asarray(img[:,:,0] + img[:, :, 1] * 256, dtype=np.float32)
        return depth

    def make_dataset(self):
        assert self.phase in ['train', 'test']

        data_path = '{}/{}'.format(self.root, self.phase)
        label_path = '{}/joint_data.mat'.format(data_path)
        center_path = '{}/center_{}_refined.txt'.format(self.root, self.phase)

        data = sorted(glob(data_path + '/depth_1*.png'))
        labels = sio.loadmat(label_path)
        labels_uvd = labels['joint_uvd'][0][:, JOINT, :][:, EVAL, :]
        labels_xyz = labels['joint_xyz'][0][:, JOINT, :][:, EVAL, :]
        center_refined_xyz = np.loadtxt(center_path)
        item = list(zip(data, labels_uvd, labels_xyz, center_refined_xyz))

        return item

