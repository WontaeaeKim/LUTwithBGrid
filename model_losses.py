import torch
import torch.nn as nn

import kornia


class TV_3D(nn.Module):
    def __init__(self, dim=33, batch_size=1):
        super(TV_3D,self).__init__()

        self.weight_r = torch.ones(3,dim,dim,dim-1, dtype=torch.float)
        self.weight_r[:,:,:,(0,dim-2)] *= 2.0
        self.weight_g = torch.ones(3,dim,dim-1,dim, dtype=torch.float)
        self.weight_g[:,:,(0,dim-2),:] *= 2.0
        self.weight_b = torch.ones(3,dim-1,dim,dim, dtype=torch.float)
        self.weight_b[:,(0,dim-2),:,:] *= 2.0
        self.relu = torch.nn.ReLU()

    def forward(self, LUT):
        dif_r = LUT[:,:,:,:,:-1] - LUT[:,:,:,:,1:]
        dif_g = LUT[:,:,:,:-1,:] - LUT[:,:,:,1:,:]
        dif_b = LUT[:,:,:-1,:,:] - LUT[:,:,1:,:,:]
        
        tv = torch.mean(torch.mul((dif_r ** 2), self.weight_r)) + torch.mean(
            torch.mul((dif_g ** 2), self.weight_g)) + torch.mean(torch.mul((dif_b ** 2), self.weight_b))

        mn = torch.mean(self.relu(dif_r)) + torch.mean(self.relu(dif_g)) + torch.mean(self.relu(dif_b))

        return tv, mn


class DeltaE_loss(nn.Module):
    def  __init__(self):
        super(DeltaE_loss,self).__init__()
    def forward(self, img, gt):
        img_lab =  kornia.color.rgb_to_lab(img)
        gt_lab = kornia.color.rgb_to_lab(gt)
        
        #img_lab = lab_normalize(img_lab)
        #gt_lab = lab_normalize(gt_lab)        
        
        img_c = torch.sqrt(torch.square(img_lab[:,1]) + torch.square(img_lab[:,2]) + 1e-12)
        gt_c = torch.sqrt(torch.square(gt_lab[:,1]) + torch.square(gt_lab[:,2]) + 1e-12)
        
        sc = 1 + 0.045*img_c
        sh_2 = (1 + 0.015*img_c)**2
  
        dc = img_c - gt_c
        dh_2 = torch.square(img_lab[:,1] - gt_lab[:,1]) + torch.square(img_lab[:,2] - gt_lab[:,2]) - torch.square(dc)
        
        loss = torch.square(img_lab[:,0] - gt_lab[:,0]) + torch.square(dc/sc) + dh_2/sh_2
        loss = torch.sqrt(torch.clamp(loss, 0, torch.inf) + 1e-12).mean()
            
        
        return loss