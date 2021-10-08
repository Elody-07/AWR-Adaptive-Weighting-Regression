import os
import os.path as osp
from tqdm import tqdm
import numpy as np

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, MultiStepLR
from torchnet import meter

from model.hourglass import PoseNet
from model.resnet_deconv import get_deconv_net
from model.loss import My_SmoothL1Loss
from dataloader.nyu_loader import NYU
from util.feature_tool import FeatureModule
from util.eval_tool import EvalUtil
from util.vis_tool import VisualUtil
from util.util import xyz2uvd, uvd2xyz

class Trainer(object):
    def __init__(self, config):
        # torch.cuda.set_device(config.gpu_id)
        cudnn.benchmark = True

        self.config = config
        self.data_dir = osp.join(self.config.data_dir, self.config.dataset)

        # output dirs for model, log and result figure saving
        self.work_dir = osp.join(self.config.output_dir, self.config.dataset, 'checkpoint_'+ self.config.exp_id)
        if not osp.exists(self.work_dir):
            os.makedirs(self.work_dir)

        self.log_file = osp.join(self.work_dir, '%s_%s.log' % (self.config.net, self.config.log_id))
        self.log = open(self.log_file, 'a')
        self.vis_tool = VisualUtil(self.config.dataset)

        # save config file
        print('-------------------start programming-------------------', file=self.log)
        for k, v in self.config.__class__.__dict__.items():
            if not k.startswith('_'):
                print(str(k) + ":" + str(v))
                print(str(k) + ":" + str(v), file=self.log)

        if 'resnet' in self.config.net:
            net_layer = int(self.config.net.split('_')[1])
            self.net = get_deconv_net(net_layer, self.config.jt_num, self.config.downsample)
        elif 'hourglass' in self.config.net:
            self.stacks = int(self.config.net.split('_')[1])
            print('hourglass stacks:', self.stacks)
            print('hourglass stacks:', self.stacks, file=self.log)
            self.net = PoseNet(self.config.net, self.config.jt_num)
        self.net = self.net.cuda()

        # init dataset, you can add other datasets
        if self.config.dataset == 'nyu':
            self.trainData = NYU(self.data_dir, 'train', img_size=self.config.img_size, aug_para=self.config.augment_para, cube=self.config.cube)

        # init optimizer
        if self.config.optimizer == 'adam':
            self.optimizer = Adam(self.net.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)
        elif self.config.optimizer == 'sgd':
            self.optimizer = SGD(self.net.parameters(), lr=self.config.lr, momentum=0.9, weight_decay=self.config.weight_decay)

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
            if 'best_records' in pth:
                self.best_records= pth['best_records']

        # init scheduler
        if self.config.scheduler == 'auto':
            self.scheduler = ReduceLROnPlateau(self.optimizer, "min", patience=2, min_lr=1e-8)
        elif self.config.scheduler == 'step':
            self.scheduler = StepLR(self.optimizer, step_size=self.config.step, gamma=0.1, last_epoch=self.best_records['epoch'])

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.config.lr
            print('learning rate: %.8f' % param_group['lr'])
            print('learning rate: %.8f' % param_group['lr'], file=self.log)

    def train(self):
        self.trainLoader = DataLoader(self.trainData, batch_size=self.config.batch_size, shuffle=True, num_workers=self.config.num_workers)

        # step4: meters
        eval_tool = EvalUtil(self.trainData.img_size, self.trainData.paras, self.trainData.flip, self.trainData.jt_num)
        loss_meter = meter.AverageValueMeter()
        # train
        for epoch in range(self.best_records['epoch']+1, self.config.max_epoch+1):
            self.net.train()
            for ii, (img, jt_xyz_gt, jt_uvd_gt, center_xyz, M, cube) in tqdm(enumerate(self.trainLoader)):
                # train model
                input = img.cuda()
                loss = 0
                self.ft_sz = int(self.config.img_size / self.config.downsample)
                if self.config.system == 'uvd':
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


                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                # meters_update
                loss_meter.add(loss.item())

                if (ii + 1) % self.config.print_freq == 0:
                    # loss_meter.value() funtion returns (loss.mean, loss.std)
                    print('[epoch: %d][train loss: %.5f][offset_loss: %.5f][coord_loss: %.5f]'
                          % (epoch,loss_meter.value()[0],loss_offset.item(),loss_coord.item()))
                    print('[epoch: %d][train loss: %.5f][offset_loss: %.5f][coord_loss: %.5f]'
                          % (epoch,loss_meter.value()[0],loss_offset.item(),loss_coord.item()),
                          file=self.log)
                    loss_meter.reset()


                jt_uvd_gt = jt_uvd_gt.detach().cpu().numpy()
                jt_xyz_gt = jt_xyz_gt.detach().cpu().numpy()
                center_xyz = center_xyz.detach().cpu().numpy()
                M = M.detach().numpy()
                cube = cube.detach().numpy()
                if self.config.system == 'uvd':
                    jt_uvd_pred = jt_uvd_pred.detach().cpu().numpy()
                    for i in range(jt_uvd_pred.shape[0]):
                        eval_tool.feed(jt_uvd_pred[i],jt_xyz_gt[i],center_xyz[i],M[i],cube[i])

            train_mpe, _, _, _, _= eval_tool.get_measures()
            print(
                "[epoch %d], [train loss %.5f], [train mpe %.5f], [lr %.8f]"
                % (epoch, loss_meter.value()[0], train_mpe, self.optimizer.param_groups[0]["lr"])
            )
            print(
                "[epoch %d], [train loss %.5f], [train mpe %.5f], [lr %.8f]"
                % (epoch, loss_meter.value()[0], train_mpe, self.optimizer.param_groups[0]["lr"]),
                file=self.log
            )

            if self.config.scheduler == 'auto':
                self.scheduler.step(train_mpe)
            elif self.config.scheduler == 'step':
                self.scheduler.step(epoch)

            # temporary save in case there is no improvement

            if epoch == self.config.max_epoch:
                save = {
                    'model': self.net.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'best_records': self.best_records
                }
                torch.save(
                    save,
                    osp.join(self.work_dir, 'epoch_{}.pth'.format(epoch))
                )
            
        self.log.flush()
        self.log.close()


if __name__=='__main__':
    from config import opt
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    trainer = Trainer(opt)
    trainer.train()