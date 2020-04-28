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

        self.log_file = osp.join(self.config.output_dir, self.config.dataset, 'full.log' )
        self.log = open(self.log_file, 'a')
        self.vis_tool = VisualUtil(self.config.dataset)

        # save config file
        print('-------------------start programming-------------------', file=self.log)
        for k, v in self.config.__class__.__dict__.items():
            if not k.startswith('_'):
                print(str(k) + ":" + str(v), file=self.log)
                    
        net_layer = int(self.config.net.split('_')[1])
        self.net = get_deconv_net(net_layer, self.config.jt_num*4, self.config.downsample)
        self.net.init_weights()

        # init dataset, you can add other datasets 
        if self.config.dataset == 'nyu':
            self.trainData = NYU(self.data_dir, 'train', img_size=self.config.img_size, aug_para=self.config.augment_para, cube=self.config.cube)
            self.valData = NYU(self.data_dir, 'train', val=True, img_size=self.config.img_size, cube=self.config.cube)
            self.testData = NYU(self.data_dir, 'test', img_size=self.config.img_size, cube=self.config.cube)

        # init optimizer
        self.optimizer = Adam(self.net.parameters(), lr=self.config.lr)

        # init scheduler
        self.scheduler = ReduceLROnPlateau(self.optimizer, "min", patience=2, min_lr=1e-8)

        # init loss function
        self.criterion = My_SmoothL1Loss().cuda()

        self.FM = FeatureModule()
        self.best_records={'epoch': 0,
                           'MPE': 1e10,
                           'AUC': 0}

        # load model
        if self.config.load_model :
            print('loading model from %s' % self.config.load_model)
            pth = torch.load(self.config.load_model)
            self.net.load_state_dict(pth['model'])
            self.optimizer.load_state_dict(pth['optimizer'])
            self.best_records= pth['best_records']
        self.net = self.net.cuda()


    def train(self):
        self.trainLoader = DataLoader(self.trainData, batch_size=self.config.batch_size, shuffle=True, num_workers=self.config.num_workers)

        # step4: meters
        loss_meter = meter.AverageValueMeter()
        # train
        for epoch in range(self.best_records['epoch']+1, self.config.max_epoch+1):
            self.net.train()
            for ii, (img, jt_xyz_gt, jt_uvd_gt, center_xyz, M, cube) in tqdm(enumerate(self.trainLoader)):

                # train model
                input = img.cuda()
                # jt_xyz_gt = jt_xyz_gt.cuda()
                jt_uvd_gt = jt_uvd_gt.cuda()
                offset_gt = self.FM.joint2offset(jt_uvd_gt, input, self.config.kernel_size, int(self.config.img_size / self.config.downsample))

                offset_pred = self.net(input)
                jt_uvd_pred = self.FM.offset2joint_softmax(offset_pred, input, self.config.kernel_size)

                loss_offset = self.config.dense_weight * self.criterion(offset_pred, offset_gt)
                loss_coord = self.config.coord_weight * self.criterion(jt_uvd_pred, jt_uvd_gt)
                loss =  loss_offset + loss_coord

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # meters_update
                loss_meter.add(loss.item())
                if (ii + 1) % self.config.print_freq == 0:
                    # loss_meter.value() funtion returns (loss.mean, loss.std)
                    print('[epoch: %d][train loss: %.3f][offset_loss: %.3f][coord_loss: %.3f]'
                          % (epoch,loss_meter.value()[0],loss_offset.item(),loss_coord.item()))
                    print('[epoch: %d][train loss: %.3f][offset_loss: %.3f][coord_loss: %.3f]'
                          % (epoch,loss_meter.value()[0],loss_offset.item(),loss_coord.item()),
                          file=self.log)
                    loss_meter.reset()

            mpe = self.val(epoch)

            self.scheduler.step(mpe)

            # temporary save in case there is no improvement

            if epoch == self.config.max_epoch:
                mpe = self.test(epoch)
                save = {
                    'model': self.net.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'best_records': self.best_records
                }
                torch.save(
                    save,
                    osp.join(self.model_save, '%s_latest_test%.3f.pth' % (self.config.net,mpe))
                )

    @torch.no_grad()
    def val(self, epoch):
        '''
        计算模型验证集上的准确率
        '''
        self.valLoader = DataLoader(self.valData, batch_size=self.config.batch_size, shuffle=False, num_workers=self.config.num_workers)
        self.net.eval()

        eval_tool = EvalUtil(self.valData.img_size, self.valData.paras, self.valData.flip, self.valData.jt_num)
        loss_meter = meter.AverageValueMeter()
        for ii, (img, jt_xyz_gt, jt_uvd_gt, center_xyz, M, cube) in tqdm(enumerate(self.valLoader)):

            input = img.cuda()
            # jt_xyz_gt = jt_xyz_gt.cuda()
            jt_uvd_gt = jt_uvd_gt.cuda()
            offset_gt = self.FM.joint2offset(jt_uvd_gt, input, self.config.kernel_size, int(self.config.img_size / self.config.downsample))

            offset_pred = self.net(input)
            jt_uvd_pred = self.FM.offset2joint_softmax(offset_pred, input, self.config.kernel_size)

            loss_offset = self.config.dense_weight * self.criterion(offset_pred, offset_gt)
            loss_coord = self.config.coord_weight * self.criterion(jt_uvd_pred, jt_uvd_gt)
            loss =  loss_offset + loss_coord
            loss_meter.add(loss.item())

            jt_uvd_gt = jt_uvd_gt.detach().cpu().numpy()
            jt_uvd_pred = jt_uvd_pred.detach().cpu().numpy()
            jt_xyz_gt = jt_xyz_gt.detach().cpu().numpy()
            center_xyz = center_xyz.detach().cpu().numpy()
            M = M.detach().numpy()
            cube = cube.detach().numpy()
            for i in range(jt_uvd_gt.shape[0]):
                eval_tool.feed(jt_uvd_pred[i],jt_xyz_gt[i],center_xyz[i],M[i],cube[i])

            if ii % self.config.vis_freq == 0:
                img = img.detach().cpu().numpy()
                path = osp.join(self.result_dir, 'val_epoch%d_iter%d.png'%(epoch, ii))
                jt_uvd_pred = (jt_uvd_pred + 1) * self.config.img_size / 2.
                jt_uvd_gt = (jt_uvd_gt + 1) * self.config.img_size / 2.
                self.vis_tool.plot(img[0], path, jt_uvd_pred[0], jt_uvd_gt[0])

        mpe, mid, auc, pck, thresh = eval_tool.get_measures()

        if mpe < self.best_records['MPE']:
            eval_tool.plot_pck(osp.join(self.result_dir, 'pck_epoch%d.png' % epoch), pck, thresh)
            mpe_test= self.test(epoch)
            self.best_records = {'epoch': epoch,
                                 'MPE': mpe,
                                 'AUC': auc}

            pth_name = "%s_epoch%d_val%.3f_test%.3f.pth" % (self.config.net, epoch, mpe, mpe_test)
            save = {
                'model': self.net.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_records': self.best_records
            }
            torch.save(save, osp.join(self.model_save, pth_name))

        print(
            "[epoch %d], [val loss %.5f], [mpe %.5f], [lr %.8f]"
            % (epoch, loss_meter.value()[0], mpe, self.optimizer.param_groups[0]["lr"])
        )
        print(
            "[epoch %d], [val loss %.5f], [mpe %.5f], [lr %.8f]"
            % (epoch, loss_meter.value()[0], mpe, self.optimizer.param_groups[0]["lr"]),
            file=self.log
        )

        print(
            "best record: [epoch %d][MPE %.3f][AUC %.3f]"
            % (self.best_records['epoch'],
               self.best_records['MPE'],
               self.best_records['AUC'])
        )
        print(
            "best record: [epoch %d][MPE %.3f][AUC %.3f]"
            % (self.best_records['epoch'],
               self.best_records['MPE'],
               self.best_records['AUC']),
            file=self.log
        )


        self.net.train()
        return mpe

    @torch.no_grad()
    def test(self, epoch=0):
        '''
        计算模型测试集上的准确率
        '''
        self.testLoader = DataLoader(self.testData, batch_size=self.config.batch_size, shuffle=False, num_workers=8)
        self.net.eval()

        eval_tool = EvalUtil(self.valData.img_size, self.valData.paras, self.valData.flip, self.valData.jt_num)
        loss_meter = meter.AverageValueMeter()
        for ii, (img, jt_xyz_gt, jt_uvd_gt, center_xyz, M, cube) in tqdm(enumerate(self.testLoader)):

            input = img.cuda()
            # jt_xyz_gt = jt_xyz_gt.cuda()
            jt_uvd_gt = jt_uvd_gt.cuda()
            offset_gt = self.FM.joint2offset(jt_uvd_gt, input, self.config.kernel_size, int(self.config.img_size / self.config.downsample))

            offset_pred = self.net(input)
            jt_uvd_pred = self.FM.offset2joint_softmax(offset_pred, input, self.config.kernel_size)

            loss_offset = self.config.dense_weight * self.criterion(offset_pred, offset_gt)
            loss_coord = self.config.coord_weight * self.criterion(jt_uvd_pred, jt_uvd_gt)
            loss =  loss_offset + loss_coord
            loss_meter.add(loss.item())

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

        print(
            "[test_loss %.5f], [mean_Error %.5f]"
            % (loss_meter.value()[0], mpe)
        )
        print(
            "[test_loss %.5f], [mean_Error %.5f]"
            % (loss_meter.value()[0], mpe),
            file=self.log
        )

        self.net.train()
        return mpe

if __name__=='__main__':
    trainer = Trainer(opt)
    trainer.train()
    # Trainer.test(-1)