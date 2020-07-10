import numpy as np
import torch
from numpy import linalg
from torch.nn import Module, Conv2d, Parameter, Softmax
import torch.nn.functional as F
import sys
import cv2

# generate dense offsets feature 
class FeatureModule:

    def joint2offset(self, jt_uvd, img, kernel_size, feature_size):
        '''
        :params joint: hand joint coordinates, shape (B, joint_num, 3)
        :params img: depth image, shape (B, C, H, W)
        :params kernel_size
        :params feature_size: size of generated offsets feature 
        '''
        batch_size, jt_num, _ = jt_uvd.size()
        img = F.interpolate(img, size = [feature_size, feature_size])
        jt_ft = jt_uvd.view(batch_size, -1, 1, 1).repeat(1, 1, feature_size, feature_size) # (B, joint_num*3, F, F)

        mesh_x = 2.0 * (torch.arange(feature_size).unsqueeze(0).expand(feature_size, feature_size).float() + 0.5) / feature_size - 1.0
        mesh_y = 2.0 * (torch.arange(feature_size).unsqueeze(1).expand(feature_size, feature_size).float() + 0.5) / feature_size - 1.0
        coords = torch.stack((mesh_x, mesh_y), dim=0)
        coords = coords.unsqueeze(0).repeat(batch_size, 1, 1, 1).to(jt_uvd.device) # (B, 2, F, F)
        coords = torch.cat((coords, img), dim=1).repeat(1, jt_num, 1, 1) # (B, jt_num*3, F, F)

        offset = jt_ft - coords # (B, jt_num*3, F, F)
        offset = offset.view(batch_size, jt_num, 3, feature_size, feature_size) # (B, jt_num, 3, F, F)
        dis = torch.sqrt(torch.sum(torch.pow(offset, 2), dim=2) + 1e-8) # (B, jt_num, F, F)

        offset_norm = offset / dis.unsqueeze(2) # (B, jt_num, 3, F, F)
        heatmap = (kernel_size - dis) / kernel_size # (B, jt_num, F, F)
        mask = heatmap.ge(0).float() * img.lt(0.99).float() # (B, jt_num, F, F)

        offset_norm_mask =  (offset_norm * mask.unsqueeze(2)).view(batch_size, -1, feature_size, feature_size)
        heatmap_mask = heatmap * mask.float()
        return torch.cat((offset_norm_mask, heatmap_mask), dim=1).float()

    def offset2joint_softmax(self, offset, img, kernel_size):
        batch_size, feature_num, feature_size, _ = offset.size()
        jt_num = int(feature_num / 4)
        img = F.interpolate(img, size = [feature_size, feature_size]) # (B, 1, F, F)
        # unit directional vector
        offset_vec = offset[:, :jt_num*3].contiguous() # (B, jt_num*3, F, F)
        # closeness heatmap
        offset_ht = offset[:, jt_num*3:].contiguous() # (B, jt_num, F, F)

        mesh_x = 2.0 * (torch.arange(feature_size).unsqueeze(0).expand(feature_size, feature_size).float() + 0.5) / feature_size - 1.0
        mesh_y = 2.0 * (torch.arange(feature_size).unsqueeze(1).expand(feature_size, feature_size).float() + 0.5) / feature_size - 1.0
        coords = torch.stack((mesh_x, mesh_y), dim=0)
        coords = coords.unsqueeze(0).repeat(batch_size, 1, 1, 1).to(offset.device)
        coords = torch.cat((coords, img), dim=1).repeat(1, jt_num, 1, 1) # (B, jt_num*3, F, F)
        coords = coords.view(batch_size, jt_num, 3, -1) # (B, jt_num, 3, F*F)

        mask = img.lt(0.99).float() # (B, 1, F, F)
        offset_vec_mask = (offset_vec * mask).view(batch_size, jt_num, 3, -1) # (B, jt_num, 3, F*F)
        offset_ht_mask =  (offset_ht * mask).view(batch_size, jt_num, -1) # (B, jt_num, F*F)
        offset_ht_norm = F.softmax(offset_ht_mask * 30, dim = -1) # (B, jt_num, F*F)
        dis = kernel_size - offset_ht_mask * kernel_size # (B, jt_num, F*F)

        jt_uvd = torch.sum((offset_vec_mask * dis.unsqueeze(2) + coords) * offset_ht_norm.unsqueeze(2), dim=-1)

        return jt_uvd.float()


