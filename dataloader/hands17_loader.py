from loader import loader
from loader import uvd2xyz, xyz2uvd, transformPoints2D
from loader import get_center_adopt


import os
import cv2
import numpy as np
from PIL import Image
import scipy.io as sio
import torch
from glob import glob

joint_select = np.array([0, 1, 3, 5, 6, 7, 9, 11, 12, 13, 15, 17, 18, 19, 21, 23, 24, 25, 27, 28, 32, 30, 31])
calculate = [0, 2, 4, 6, 8, 10, 12, 14, 16, 17, 18, 21, 22, 20]


def draw(img, pts):
    img_ = img.copy()
    pts_ = pts.copy()
    pts_ = pts_.astype(np.int16)
    for pt in pts_:
        # print(pt)
        cv2.circle(img_, (pt[0], pt[1]), 1, (255, 255, 255), -1)
    return img_


def hands17_reader(img_path):
    img = Image.open(img_path)  # open image
    assert len(img.getbands()) == 1  # ensure depth image
    depth = np.asarray(img, np.float32)
    return depth



class hands17_loader(loader):

    def __init__(self, root, phase, val=False, img_size=128, center_type='refine', aug_para=[10, 0.1, 180],
                 cube_size=[200, 200, 200], joint_num=21, load_num = 957032):
        super(hands17_loader, self).__init__(root, phase, img_size, center_type, 'hands17')
        self.val = val
        self.flip = 1  # flip y axis when doing xyz <-> uvd transformation
        self.paras = (475.065948, 475.065857, 315.944855, 245.287079)
        self.cube_size = np.asarray(cube_size)

        self.load_num = load_num
        self.joint_num = joint_num
        self.aug_para = aug_para

        self.data = self.make_dataset(root, phase, val)
        print("loading dataset, containing %d images." % len(self.data))

    def __getitem__(self, index):
        img_path = self.data[index][0]
        depth = hands17_reader(img_path)

        jt_uvd = self.data[index][1].copy()
        jt_xyz = self.data[index][2].copy()
        cube_size = self.cube_size
        center_refined_xyz = self.data[index][3].copy()

        if self.center_type == 'mass':
            center_uvd = get_center_adopt(depth)
            center_xyz = uvd2xyz(center_uvd, self.paras, self.flip)
        elif self.center_type == 'random':
            random_trans = (np.random.rand(3) - 0.5) * 0.4 * cube_size
            center_xyz = jt_xyz.mean(0) + random_trans
            center_uvd = xyz2uvd(center_xyz, self.paras, self.flip)
        elif self.center_type == 'joint':
            center_xyz = jt_xyz[20]
            center_uvd = xyz2uvd(center_xyz, self.paras, self.flip)
        elif self.center_type == 'joint_mean':
            center_xyz = jt_xyz.mean(0)
            center_uvd = xyz2uvd(center_xyz, self.paras, self.flip)
        else:
            center_xyz = center_refined_xyz
            center_uvd = xyz2uvd(center_xyz, self.paras, self.flip)

        jt_xyz_crop = jt_xyz - center_xyz
        depth_crop, M = self.CropHand(depth, center_uvd, cube_size, dsize=(self.img_size, self.img_size))

        if self.phase == 'train' and self.val == False:
            aug_op, trans, scale, rot = self.RandomAug(sigma_trans=self.aug_para[0], sigma_scale=self.aug_para[1],
                                                       rot_range=self.aug_para[2])
            # print(aug_op)
            new_img, new_jt_xyz, new_cube, new_center_uvd, newM = self.AugmentHand(depth_crop, jt_xyz_crop, center_uvd,
                                                                                   self.cube_size, M, aug_op, trans,
                                                                                   scale, rot)
            new_jt_xyz = new_jt_xyz / (new_cube[2] / 2.)
        else:
            new_img = self.Normalize_img(depth_crop.max(), depth_crop, center_xyz, cube_size)
            new_jt_xyz = jt_xyz_crop / (cube_size[2] / 2.)
            new_cube = np.array(cube_size)
            new_center_uvd = center_uvd
            newM = M

        new_center_xyz = uvd2xyz(new_center_uvd, self.paras, self.flip)
        new_jt_uvd = transformPoints2D(
            xyz2uvd(new_jt_xyz * (new_cube[2] / 2.0) + new_center_xyz, self.paras, self.flip), newM)
        new_jt_uvd[:, :2] = new_jt_uvd[:, :2] / (self.img_size / 2.) - 1
        new_jt_uvd[:, 2] = (new_jt_uvd[:, 2] - new_center_xyz[2]) / (new_cube[0] / 2.0)

        data = torch.from_numpy(new_img).float()
        data = data.unsqueeze(0)
        j3d_xyz = torch.from_numpy(new_jt_xyz).float()
        j3d_uvd = torch.from_numpy(new_jt_uvd).float()
        center = torch.from_numpy(new_center_xyz).float()
        newM = torch.from_numpy(newM).float()
        cube = torch.from_numpy(new_cube).float()

        return data, j3d_xyz, j3d_uvd, center, newM, cube

    def __len__(self):
        return len(self.data)

    def make_dataset(self, root, phase, val=False):
        assert phase in ['train', 'test']

        labels_xyz, labels_uvd, center_refined_xyz, center_refined_uvd, data = self.read_joints(root, phase)

        item = list(zip(data, labels_uvd, labels_xyz, center_refined_xyz))

        if phase == 'test':
            return item
        elif phase == 'train' and val:
            return [x for index, x in enumerate(item) if index % 10 == 0]
        else:
            return [x for index, x in enumerate(item) if index % 10 != 0]

    def read_joints(self, data_rt, img_type):
        centers_xyz = []
        centers_uvd = []
        joints_xyz = []
        joints_uvd = []
        img_names = []
        if img_type == 'train':
            f = open(data_rt + "/training/Training_Annotation.txt", "r")
            f_center = open(data_rt + "/center_train_refined.txt")
            lines = f.readlines()
            lines_center = f_center.readlines()
            f.close()
            f_center.close()

            for index, line in enumerate(lines):
                if index > self.load_num:
                    break
                strs = line.split()
                img_path = data_rt + '/training/images/' + strs[0]
                if not os.path.isfile(img_path):
                    continue

                joint_xyz = np.array(list(map(float, strs[1:]))).reshape(-1, 3)
                strs_center = lines_center[index].split()

                if strs_center[0] == 'invalid':
                    continue
                else:
                    center_xyz = np.array(list(map(float, strs_center))).reshape(3)
                center_uvd = xyz2uvd(center_xyz, self.paras, self.flip)
                centers_xyz.append(center_xyz)
                centers_uvd.append(center_uvd)

                joint_uvd = xyz2uvd(joint_xyz, self.paras, self.flip)
                joints_xyz.append(joint_xyz)
                joints_uvd.append(joint_uvd)
                img_names.append(img_path)
        else:
            f = open(data_rt + "/frame/BoundingBox.txt", "r")
            f_center = open(data_rt + "/center_test_refined.txt")
            lines = f.readlines()
            lines_center = f_center.readlines()
            f.close()
            f_center.close()
            for index, line in enumerate(lines):
                if index > self.load_num:
                    break
                strs = line.split()
                img_path = data_rt + '/frame/images/' + strs[0]
                if not os.path.isfile(img_path):
                    continue
                strs_center = lines_center[index].split()
                if strs_center[0] == 'invalid':
                    continue
                else:
                    center_xyz = np.array(list(map(float, strs_center))).reshape(3)
                center_uvd = xyz2uvd(center_xyz, self.paras, self.flip)
                centers_xyz.append(center_xyz)
                centers_uvd.append(center_uvd)

                joints_xyz.append(np.ones([21, 3]))
                joints_uvd.append(np.ones([21, 3]))
                img_names.append(img_path)

        return joints_xyz, joints_uvd, centers_xyz, centers_uvd, img_names


if __name__ == "__main__":

    from torch.utils.data import DataLoader

    root = '/home/ljs/pfren/dataset/hands17'
    dataset = iter(hands17_loader(root, 'train', load_num=1000))
    data = next(dataset)  # 'trans'
    data = next(dataset)  # 'scale'
    data = next(dataset)  # 'scale'
    data = next(dataset)  # 'None'
    data = next(dataset)  # 'scale'
    data = next(dataset)  # 'trans'
    data = next(dataset)  # 'trans'
    data = next(dataset)  # 'trans'
    data = next(dataset)  # 'None'
    # data = next(dataset) # 'rot'
    img, j3d_xyz, j3d_uvd, center_xyz, M, cube = data
    for item in data:
        print(item.size())

    a = draw(img[0].numpy(), j3d_uvd[:, :2].numpy())
    # cv2.imwrite("./none_j2d.png", (a + 1) * 100)

    print('done')



