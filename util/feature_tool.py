import numpy as np
import torch
from numpy import linalg
from torch.nn import Module, Conv2d, Parameter, Softmax
import torch.nn.functional as F
import sys
import cv2

# generate dense offsets feature 
class GFM:

    def joint2offset(self, joint, img, kernel_size, feature_size):
        '''
        :params joint: hand joint coordinates, shape (B, joint_num, 3)
        :params img: depth image, shape (B, C, H, W)
        :params kernel_size
        :params feature_size: size of generated offsets feature 
        '''
        batch_size, joint_num, _ = joint.size()
        img = F.interpolate(img, size = [feature_size, feature_size])
        joint_ft = joint.view(batch_size, -1, 1, 1).repeat(1, 1, feature_size, feature_size) # (B, joint_num*3, F, F)

        mesh_x = 2.0 * torch.arange(feature_size).unsqueeze(0).expand(feature_size, feature_size).float() / (feature_size - 1.0) - 1.0
        mesh_y = 2.0 * torch.arange(feature_size).unsqueeze(1).expand(feature_size, feature_size).float() / (feature_size - 1.0) - 1.0
        coords = torch.stack((mesh_x, mesh_y), dim=0)
        coords = coords.unsqueeze(0).repeat(batch_size, 1, 1, 1).to(joint.device) # (B, 2, F, F)
        coords = torch.cat((coords, img), dim=1).repeat(1, joint_num, 1, 1) # (B, joint_num*3, F, F)

        offset = joint_ft - coords
        offset = offset.view(batch_size, joint_num, 3, feature_size, feature_size) # (B, joint_nums, 3, F, F)

        dis = torch.sqrt(torch.sum(torch.pow(offset, 2), dim=2, keepdim=True) + 1e-8)
        offset_norm = (offset / dis).view(batch_size, -1, feature_size, feature_size)

        heatmap = (kernel_size - dis) / kernel_size
        mask = heatmap.ge(0) * img.lt(0.99)
        offset_norm *=  mask
        heatmap *= mask
        return torch.cat((offset_norm, heatmap), dim=1)

    def offset2joint_softmax(self, offset, img, kernel_size):
        batch_size, feature_num, feature_size, _ = offset.size()
        joint_num = int(feature_num / 4)
        img = F.interpolate(img, size = [feature_size, feature_size])
        # unit directional vector
        offset_vec = offset[:, :joint_num*3].contiguous()
        # closeness heatmap
        offset_ht = offset[:, joint_num*3:].contiguous()

        mesh_x = 2.0 * torch.arange(feature_size).unsqueeze(0).expand(feature_size, feature_size).float() / (feature_size - 1.0) - 1.0
        mesh_y = 2.0 * torch.arange(feature_size).unsqueeze(1).expand(feature_size, feature_size).float() / (feature_size - 1.0) - 1.0
        coords = torch.stack((mesh_x, mesh_y), dim=0)
        coords = coords.unsqueeze(0).repeat(batch_size, 1, 1, 1).to(offset.device)
        coords = torch.cat((coords, img), dim=1).repeat(1, joint_num, 1, 1) # (B, joint_num*3, F, F)

        mask = img.lt(0.99).float() # (B, 1, F, F)
        offset_vec = (offset_vec * mask).view(batch_size, joint_num, 3, -1) # (B, joint_nums, 3, F*F)
        offset_ht =  (offset_ht * mask).view(batch_size, joint_num, -1) # (B, joint_nums, F*F)
        offset_ht = F.softmax(offset_ht * 30, dim = -1)
        dis = kernel_size - offset_ht * kernel_size # (B, joint_num, F*F)

        joint = torch.sum((offset_vec * dis.unsqueeze(2).repeat(1,1,3,1) + coords) * offset_ht.unsqueeze(2).repeat(1,1,3,1), dim=-1)

        return joint



if __name__ == '__main__':
    batch_size = 2
    joint_num = 21
    cuda_id = 0
    torch.cuda.set_device(cuda_id)
    GFM_ = GFM(batch_size, cuda_id, 64, joint_num)
    # test_data = ren_loader.icvl_loader('/data/users/pfren/data/dataset/hand/icvl', 'test')
    # test_data = ren_loader.nyu_loader('/data/users/ljs/pfren/dataset/nyu', 'test')
    test_data = ren_loader.msra_loader('/data/users/ljs/pfren/dataset/msra', 'test',test_persons=[0])
    from torch.utils.data import DataLoader
    dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=1)
    for index, data in enumerate(dataloader):
        img, img_ori, joint_world, joint_img, center, M, cube, pcl_normal, joint_normal, offset, coeff, max_bbx_len = data
        # offset = GFM_.joint2offset(joint_img,img,feature_size=128)
        # heatmap = offset[:,joint_num*3:,:,:]
        debug_2d_heatmap(img,joint_img,index,GFM_)
        # for i in range(img.size(0)):
        #     # for joint_id in range(joint_num):
        #         depth = img.numpy()[i]
        #         # feature = heatmap.numpy()[i,joint_id]
        #         img_heatmap = ((depth + 1) / 4 ) * 255.0
        #         color = cv2.cvtColor(np.transpose(img_heatmap, (1, 2, 0)), cv2.COLOR_GRAY2RGB)
        #         # color[:, :, 1] = 255
        #         img_name_1 = './debug/3dheatmap/' + str(index) + '_' + str(i) + '.png'
        #         cv2.imwrite(img_name_1, color)

        # joint = GFM_.offset2joint(offset,img)
        # print(index,(joint_img-joint).sum())
        # debug_offset(data,index,GFM_)
        print(index)

