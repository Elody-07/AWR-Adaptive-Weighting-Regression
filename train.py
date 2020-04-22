
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
from tensorboardX import SummaryWriter

from model.loss import Modified_SmoothL1Loss,My_SmoothL1Loss
from model.resnet_deconv import get_deconv_net
from dataloader import nyu_loader,hands17_loader
from dataloader import ren_loader
from dataloader.loader import uvd2xyz, get_trans_points
from util.generateFeature import GFM
from util.misc import uvd2error
from config import opt


class Trainer(object):
    def __init__(self, config):
        torch.cuda.set_device(self.config.gpu_id)
        cudnn.benchmark = True

        self.config = config
        self.data_rt = osp.join(self.config.root_dir, self.config.dataset)

        # output dirs for model, log and result figure saving
        self.model_save = osp.join(self.config.output_dir, self.config.dataset, 'checkpoint')
        self.result_dir = osp.join(self.config.output_dir, self.config.dataset, 'results' )
        if not osp.exists(self.model_save):
            os.makedirs(self.model_save)
        if not osp.exists(self.result_dir):
            os.makedirs(self.result_dir)

        self.log_file = osp.join(self.config.output_dir, self.config.dataset, 'full.log' )
        self.log = open(self.log_file, 'a')

        
        # save config file
        for k, v in self.config.__class__.__dict__.items():
            if not k.startswith('_'):
                print('-------------------start programming-------------------', log=self.log)
                print(str(k) + ":" + str(v), log=self.log)
                    
        net_layer = int(self.config.net.split('_')[1])
        self.net = get_deconv_net(net_layer, self.config.joint_num*4, self.config.downsample)
        self.net.init_weights()


        # init dataset, you can add other datasets 
        if self.config.dataset == 'nyu':
            self.trainData = nyu_loader.nyu_loader(self.data_rt, 'train', aug_para=self.config.augment_para, img_size=self.config.input_size, cube_size=self.config.cube_size, center_type=self.config.center_type)

            self.valData = nyu_loader.nyu_loader(self.data_rt, 'train', val=True, img_size=self.config.input_size, cube_size=self.config.cube_size, center_type=self.config.center_type)

            self.testData = nyu_loader.nyu_loader(self.data_rt, 'test', img_size=self.config.input_size, cube_size=self.config.cube_size, center_type=self.config.center_type)


        # init optimizer
        self.optimizer = Adam(self.net.parameters(), lr=self.config.lr)

        # init scheduler
        self.scheduler = ReduceLROnPlateau(self.optimizer, "min", patience=2, min_lr=1e-8)

        # init loss function
        self.criterion = My_SmoothL1Loss().cuda()

        self.GFM_ = GFM()
        self.best_records={}
        self.best_records['epoch'] = 0
        self.best_records['val_loss'] = 1e10 
        self.best_records['error'] = 1e10 

        # load model
        if self.config.load_model :
            print('loading model from %s' % self.config.load_model)
            pth = torch.load(self.config.load_model)
            self.net.load_state_dict(pth['model'])
            self.optimizer.load_state_dict(pth['optimizer'])
            self.best_records= pth['best_records']

        self.net = self.net.cuda()


    def train(self):
        self.trainLoader = DataLoader(self.trainData, batch_size=self.config.batch_size, shuffle=True, num_workers=0)

        # step4: meters
        loss_meter = meter.AverageValueMeter()
        error_meter = meter.AverageValueMeter()
        # train
        for epoch in range(self.best_records['epoch'], self.config.max_epoch):
            self.net.train()
            for ii, (img, xyz_gt, uvd_gt, center, M, cube) in tqdm(enumerate(self.trainLoader)):

                # train model
                input = img.cuda()
                uvd_gt = uvd_gt.cuda()
                offset_gt = self.GFM_.joint2offset(uvd_gt, input, self.config.kernel_size, int(self.config.input_size / self.config.downsample))

                self.optimizer.zero_grad()

                offset_pd = self.net(input)
                joint_pd = self.GFM_.offset2joint(offset_pd, input, self.config.kernel_size)

                loss_offset = self.criterion(offset_pd, offset_gt) * self.config.deconv_weight
                loss_coord = self.criterion(joint_pd, uvd_gt) * self.config.coord_weight

                loss = loss_offset + loss_coord
                loss.backward()

                self.optimizer.step()
                error, _ = uvd2error(joint_pd.detach(), xyz_gt, center, M, cube, self.config.input_size, self.trainData.paras, self.trainData.flip)
                # meters_update
                loss_meter.add(loss.item())
                error_meter.add(error.item())
                # loss_meter.value()[0] * 100
                self.writer.add_scalar("train/train_map_loss", loss_offset.item(),
                                      epoch * len(self.trainLoader) + ii)
                self.writer.add_scalar("train/train_coord_loss", loss_coord.item(),
                                      epoch * len(self.trainLoader) + ii)
                self.writer.add_scalar("train/train_error", error.item(),
                                      epoch * len(self.trainLoader) + ii)
                if (ii + 1) % self.config.print_freq == 0:
                    # loss_meter.value() funtion returns (loss.mean, loss.std)
                    print(
                        "[epoch %d], [iter %d / %d], [avg train loss %.5f],[avg train error %.5f]"
                        % (epoch, ii + 1, len(self.trainLoader), loss_meter.value()[0], error_meter.value()[0])
                    )
                    loss_meter.reset()
                    error_meter.reset()
            val_error = self.val(epoch)

            if self.config.scheduler == 'auto':
                self.scheduler.step(val_error)
            elif self.config.scheduler == 'step':
                self.scheduler.step(epoch)

            # temporary save in case there is no improvement

            if (epoch + 1) == self.config.max_epoch:
                test_error = self.test(epoch)
                save = {
                    "model": self.net.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "epoch": epoch
                }
                torch.save(
                    save,
                    self.model_dir + "/latest.pth"
                )

    @torch.no_grad()
    def val(self, epoch):
        '''
        计算模型验证集上的准确率
        '''
        self.valLoader = DataLoader(self.valData, batch_size=self.config.batch_size, shuffle=False, num_workers=8)
        val_loss = meter.AverageValueMeter()
        self.net.eval()

        Error = []

        for ii, (img, xyz_gt, uvd_gt, center, M, cube) in tqdm(enumerate(self.valLoader)):
            input = img.cuda()
            uvd_gt = uvd_gt.cuda()
            offset_gt = self.GFM_.joint2offset(uvd_gt, input, self.config.kernel_size, int(self.config.input_size/self.config.downsample))

            offset_pd = self.net(input)
            joint_pd = self.GFM_.offset2joint(offset_pd, input, self.config.kernel_size)

            loss_offset = self.criterion(offset_pd, offset_gt) * self.config.deconv_weight
            loss_coord = self.criterion(joint_pd, uvd_gt) * self.config.coord_weight

            loss = loss_offset + loss_coord

            val_loss.add(loss.item())

            error, _ = uvd2error(joint_pd, xyz_gt, center, M, cube, self.config.input_size, self.trainData.paras, self.trainData.flip)
            Error.append(error.cpu().numpy())

        mean_Error = np.mean(Error)

        if mean_Error < self.best_records["Error"]:
            test_Error = self.test(epoch)
            self.best_records["epoch"] = epoch
            self.best_records["val_loss"] = val_loss.value()[0]
            self.best_records["Error"] = mean_Error

            pth_name = "best_eval_%.3f_test_%.3f" % (mean_Error, test_Error)
            save = {
                "model": self.net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "best_records": self.best_records
            }
            torch.save(save, self.model_dir + '/' + pth_name + ".pth")

        print(
            "-----------------------------------------------------------------------------------------------------------"
        )
        print(
            "[epoch %d], [val loss %.5f], [mean_Error %.5f], [lr %.8f]"
            % (epoch, val_loss.value()[0], mean_Error, self.optimizer.param_groups[0]["lr"])
        )

        print(
            "best record: [val loss %.5f], [mean_Error %.5f], [epoch %d]"
            % (
                self.best_records["val_loss"],
                self.best_records["Error"],
                self.best_records["epoch"]
            )
        )

        print(
            "-----------------------------------------------------------------------------------------------------------"
        )

        self.writer.add_scalar("train/lr", self.optimizer.param_groups[0]["lr"], epoch)
        self.writer.add_scalar("val/val_loss", val_loss.value()[0], epoch)
        self.writer.add_scalar("val/mean_Error", mean_Error, epoch)

        self.net.train()
        return mean_Error

    @torch.no_grad()
    def test(self, epoch=-1):
        '''
        计算模型测试集上的准确率
        '''
        self.testLoader = DataLoader(self.testData, batch_size=self.config.batch_size, shuffle=False, num_workers=8)
        test_loss = meter.AverageValueMeter()
        self.net.eval()

        Error = []
        joint_uvd = []

        for ii, (img, xyz_gt, uvd_gt, center, M, cube) in tqdm(enumerate(self.testLoader)):
            input = img.cuda()
            uvd_gt = uvd_gt.cuda()
            offset_gt = self.GFM_.joint2offset(uvd_gt, input, self.config.kernel_size, int(self.config.input_size/self.config.downsample))

            offset_pd = self.net(input)
            joint_pd = self.GFM_.offset2joint(offset_pd, input, self.config.kernel_size)

            loss_offset = self.criterion(offset_pd, offset_gt) * self.config.deconv_weight
            loss_coord = self.criterion(joint_pd, uvd_gt) * self.config.coord_weight

            loss = loss_offset + loss_coord

            test_loss.add(loss.item())

            error, batch_joint_uvd = uvd2error(joint_pd, xyz_gt, center, M, cube, self.config.input_size, self.trainData.paras, self.trainData.flip)
            Error.append(error.cpu().numpy())
            joint_uvd.append(batch_joint_uvd.cpu().numpy())

        mean_Error = np.mean(Error)

        txt_file = self.model_dir + '/' + 'latest_test_%.3f.txt' % mean_Error
        joint_uvd = np.vstack(joint_uvd)
        if not txt_file == None:
            np.savetxt(txt_file, joint_uvd.reshape([joint_uvd.shape[0], self.config.joint_num * 3]), fmt='%.3f')
        # if mean_Error < self.test_error:
        #     self.test_error = mean_Error
        #     pth_name = "best_test_error"
        #     save = {
        #         "model": self.net.state_dict(),
        #         "optimizer": self.optimizer.state_dict(),
        #         "epoch": epoch
        #     }
        #     torch.save(save, self.model_dir + '/' + pth_name + ".pth")

        print(
            "-----------------------------------------------------------------------------------------------------------"
        )
        print(
            "[test_loss %.5f], [mean_Error %.5f]"
            % (test_loss.value()[0], mean_Error)
        )
        print(
            "-----------------------------------------------------------------------------------------------------------"
        )
        self.writer.add_scalar("test/mean_Error", mean_Error, epoch )
        self.writer.add_scalar("test/test_loss", test_loss.value()[0], epoch)

        self.net.train()
        return mean_Error



if __name__=='__main__':
    Trainer = Trainer(opt)
    # Trainer.train()
    Trainer.test(-1)