import os
import os.path as osp
import sys
import cv2
import math
from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, MultiStepLR
from torchnet import meter

from model.resnet_deconv import get_deconv_net
from model.loss import Modified_SmoothL1Loss,My_SmoothL1Loss
from dataloader.nyu_loader import NYU
from util.feature_tool import FeatureModule
from util.eval_tool import EvalUtil
from util.vis_tool import VisualUtil
from config import opt


class Trainer(object):

    def __init__(self, config):
        torch.cuda.set_device(config.gpu_id)
        cudnn.benchmark = True

        self.config = config
        self.data_dir = osp.join(self.config.data_dir, self.config.dataset)

        # output dirs for model, log and result figure saving
        self.model_save = osp.join(self.config.output_dir, self.config.dataset, 'checkpoint')
        self.result_dir = osp.join(self.config.output_dir, self.config.dataset, 'results' )
        if not osp.exists(self.model_save):
            os.makedirs(self.model_save)
        if not osp.exists(self.result_dir):
            os.makedirs(self.result_dir)

        self.vis_tool = VisualUtil(self.config.dataset)

        net_layer = int(self.config.net.split('_')[1])
        self.net = get_deconv_net(net_layer, self.config.jt_num*4, self.config.downsample)
        self.net.init_weights()

        if self.config.dataset == 'nyu':
            self.testData = NYU(self.data_dir, 'test', img_size=self.config.img_size, cube=self.config.cube)

        self.FM = FeatureModule()

        # load model
        if self.config.load_model :
            print('loading model from %s' % self.config.load_model)
            pth = torch.load(self.config.load_model)
            self.net.load_state_dict(pth['model'])
        self.net = self.net.cuda()

    @torch.no_grad()
    def test(self, epoch=0):
        '''
        计算模型测试集上的准确率
        '''
        self.testLoader = DataLoader(self.testData, batch_size=self.config.batch_size, shuffle=False, num_workers=8)
        self.net.eval()

        eval_tool = EvalUtil(self.testData.img_size, self.testData.paras, self.testData.flip, self.testData.jt_num)
        for ii, (img, jt_xyz_gt, jt_uvd_gt, center_xyz, M, cube) in tqdm(enumerate(self.testLoader)):

            input = img.cuda()
            # jt_xyz_gt = jt_xyz_gt.cuda()

            offset_pred = self.net(input)
            jt_uvd_pred = self.FM.offset2joint_softmax(offset_pred, input, self.config.kernel_size)

            jt_uvd_gt = jt_uvd_gt.detach().cpu().numpy()
            jt_uvd_pred = jt_uvd_pred.detach().cpu().numpy()
            jt_xyz_gt = jt_xyz_gt.detach().cpu().numpy()
            center_xyz = center_xyz.detach().cpu().numpy()
            M = M.detach().numpy()
            cube = cube.detach().numpy()
            for i in range(jt_uvd_gt.shape[0]):
                eval_tool.feed(jt_uvd_pred[i],jt_xyz_gt[i],center_xyz[i],M[i],cube[i])

        mpe, mid, auc, pck, _ = eval_tool.get_measures()

        txt_file = osp.join(self.model_save, 'test_%.3f.txt' % mpe)
        jt_uvd = np.array(eval_tool.jt_uvd_pred, dtype=np.float32)
        if not txt_file == None:
            np.savetxt(txt_file, jt_uvd.reshape([jt_uvd.shape[0], self.config.jt_num * 3]), fmt='%.3f')

        print('[mean_Error %.5f]' % mpe)


if __name__=='__main__':
    Trainer.test()