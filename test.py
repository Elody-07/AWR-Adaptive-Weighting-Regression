import os
import os.path as osp
from tqdm import tqdm
import numpy as np

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchnet import meter

from model.resnet_deconv import get_deconv_net
from model.hourglass import PoseNet
from model.loss import My_SmoothL1Loss
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
        self.work_dir = osp.join(self.config.output_dir, self.config.dataset, 'checkpoint')
        self.result_dir = osp.join(self.config.output_dir, self.config.dataset, 'results' )
        if not osp.exists(self.work_dir):
            os.makedirs(self.work_dir)
        if not osp.exists(self.result_dir):
            os.makedirs(self.result_dir)

        if 'resnet' in self.config.net:
            net_layer = int(self.config.net.split('_')[1])
            self.net = get_deconv_net(net_layer, self.config.jt_num, self.config.downsample)
        elif 'hourglass' in self.config.net:
            self.stacks = int(self.config.net.split('_')[1])
            self.net = PoseNet(self.config.net, self.config.jt_num)
        self.net = self.net.cuda()

        if self.config.load_model :
            print('loading model from %s' % self.config.load_model)
            pth = torch.load(self.config.load_model)
            self.net.load_state_dict(pth['model'])
            print(pth['best_records'])
        self.net = self.net.cuda()

        if self.config.dataset == 'nyu':
            self.testData = NYU(self.data_dir, 'test', img_size=self.config.img_size, cube=self.config.cube)
        
        self.criterion = My_SmoothL1Loss().cuda()

        self.FM = FeatureModule()


    @torch.no_grad()
    def test(self, epoch):
        self.testLoader = DataLoader(self.testData, batch_size=self.config.batch_size, shuffle=False, num_workers=self.config.num_workers)
        self.net.eval()

        eval_tool = EvalUtil(self.testData.img_size, self.testData.paras, self.testData.flip, self.testData.jt_num)
        loss_meter = meter.AverageValueMeter()
        for ii, (img, jt_xyz_gt, jt_uvd_gt, center_xyz, M, cube) in tqdm(enumerate(self.testLoader)):

            input = img.cuda()
            loss = 0
            self.ft_sz = int(self.config.img_size / self.config.downsample)
            jt_uvd_gt = jt_uvd_gt.cuda()
            offset_gt = self.FM.joint2offset(jt_uvd_gt, input, self.config.kernel_size, self.ft_sz)
            if 'hourglass' in self.config.net:
                for stage_idx in range(self.stacks):
                    offset_pred = self.net(input)[stage_idx]
                    jt_uvd_pred = self.FM.offset2joint_softmax(offset_pred, input, self.config.kernel_size)
                    loss_coord = self.config.coord_weight * self.criterion(jt_uvd_pred, jt_uvd_gt)
                    loss_offset = self.config.dense_weight * self.criterion(offset_pred, offset_gt)
                    loss += (loss_coord + loss_offset)
            else:
                offset_pred = self.net(input)
                jt_uvd_pred = self.FM.offset2joint_softmax(offset_pred, input, self.config.kernel_size)
                loss_coord = self.config.coord_weight * self.criterion(jt_uvd_pred, jt_uvd_gt)
                loss_offset = self.config.dense_weight * self.criterion(offset_pred, offset_gt)
                loss += (loss_coord + loss_offset)

            loss_meter.add(loss.item())

            jt_uvd_gt = jt_uvd_gt.detach().cpu().numpy()
            jt_xyz_gt = jt_xyz_gt.detach().cpu().numpy()
            center_xyz = center_xyz.detach().cpu().numpy()
            M = M.detach().numpy()
            cube = cube.detach().numpy()
            jt_uvd_pred = jt_uvd_pred.detach().cpu().numpy()
            for i in range(jt_uvd_pred.shape[0]):
                eval_tool.feed(jt_uvd_pred[i],jt_xyz_gt[i],center_xyz[i],M[i],cube[i])

        mpe, mid, auc, pck, thresh = eval_tool.get_measures()
        print("results: [epoch %d][MPE %.3f][AUC %.3f]" % (epoch, mpe, auc))

        if epoch == -1:
            eval_tool.plot_pck(osp.join(self.result_dir, 'test_pck_epoch%d.png' % epoch), pck, thresh)

            txt_file = osp.join(self.work_dir, 'test_%.3f.txt' % mpe)
            jt_uvd = np.array(eval_tool.jt_uvd_pred, dtype = np.float32)
            if not txt_file == None:
                np.savetxt(txt_file, jt_uvd.reshape([jt_uvd.shape[0], self.config.jt_num * 3]), fmt='%.3f')

        return mpe


if __name__=='__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    trainer = Trainer(opt)
    trainer.test(-1)