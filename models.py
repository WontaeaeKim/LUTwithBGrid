import torch

import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from torchvision.transforms.functional import to_pil_image

from cpp_ext_interface import trilinear_lut_transform, tetrahedral_lut_transform
from cpp_ext_interface import trilinear_slice_function, tetrahedral_slice_function


def discriminator_block(in_filters, out_filters, normalization=False):
    """Returns downsampling layers of each discriminator block"""
    layers = [nn.Conv2d(in_filters, out_filters, 3, stride=2, padding=1)]
    layers.append(nn.LeakyReLU(0.2))
    if normalization:
        layers.append(nn.InstanceNorm2d(out_filters, affine=True))
        #layers.append(nn.BatchNorm2d(out_filters))

    return layers

class Backbone(nn.Module):
    def __init__(self, backbone_coef=8):
        super(Backbone, self).__init__()
        self.backbone_coef = backbone_coef
        self.model = nn.Sequential(
            nn.Upsample(size=(256,256),mode='bilinear'),
            nn.Conv2d(3, backbone_coef, 3, stride=2, padding=1), #8 x 128 x 128
            nn.LeakyReLU(0.2),
            nn.InstanceNorm2d(backbone_coef, affine=True),
            *discriminator_block(backbone_coef, 2*backbone_coef, normalization=True), #16 x 64 x 64
            *discriminator_block(2*backbone_coef, 4*backbone_coef, normalization=True), #32 x 32 x 32
            *discriminator_block(4*backbone_coef, 8*backbone_coef, normalization=True), #64 x 16 x 16
            *discriminator_block(8*backbone_coef, 8*backbone_coef),   #64 x 8 x 8
            #*discriminator_block(128, 128, normalization=True),
            nn.Dropout(p=0.5),
            nn.AvgPool2d(5, stride=2) #64 x 2 x 2
        )

    def forward(self, img_input):

        return self.model(img_input).view([-1,self.backbone_coef*32])

class resnet18_224(nn.Module):

    def __init__(self, aug_test=False):
        super(resnet18_224, self).__init__()

        self.aug_test = aug_test
        net = models.resnet18(pretrained=True)

        self.upsample = nn.Upsample(size=(224, 224), mode='bilinear')
        net.fc = nn.Identity()
        self.model = net
       
    def forward(self, x):
        x = self.upsample(x)
        if self.aug_test:
            x = torch.cat((x, torch.flip(x, [3])), 0)
        f = self.model(x)

        return f
    

class Gen_3D_LUT(nn.Module):
    def __init__(self, n_colors=3, n_vertices=17, n_feats=256, n_ranks=24):
        super(Gen_3D_LUT, self).__init__()
        
        # h0
        self.weights_generator = nn.Linear(n_feats, n_ranks)
        # h1
        self.basis_luts_bank = nn.Linear(
            n_ranks, n_colors * (n_vertices ** n_colors))

        self.n_colors = n_colors
        self.n_vertices = n_vertices
        self.n_feats = n_feats
        self.n_ranks = n_ranks
  
    
    def init_weights(self):
        r"""Init weights for models.

        For the mapping f (`backbone`) and h (`lut_generator`), we follow the initialization in
            [3D-LUT](https://github.com/HuiZeng/Image-Adaptive-3DLUT).

        """
        nn.init.ones_(self.weights_generator.bias)
        nn.init.zeros_(self.basis_luts_bank.bias)
        identity_lut = torch.stack([
            torch.stack(
                torch.meshgrid(*[torch.arange(self.n_vertices) for _ in range(self.n_colors)]),
                dim=0).div(self.n_vertices - 1).flip(0),
            *[torch.zeros(
                self.n_colors, *((self.n_vertices,) * self.n_colors)) for _ in range(self.n_ranks - 1)]
            ], dim=0).view(self.n_ranks, -1)
        self.basis_luts_bank.weight.data.copy_(identity_lut.t())
       
    def forward(self, img_feature):
        weights = self.weights_generator(img_feature)
        luts = self.basis_luts_bank(weights)

        luts = luts.view([-1,self.n_colors,self.n_vertices,self.n_vertices,self.n_vertices])
        return luts, weights    


class Gen_bilateral_grids(nn.Module):
    def __init__(self, n_colors=3, n_vertices=17, n_feats=256, n_ranks=24, n_grids=9):
        super(Gen_bilateral_grids, self).__init__()
        
        # h0
        self.weights_generator = nn.Linear(n_feats, n_ranks)
        # h1
        self.basis_grids_bank = nn.Linear(
            n_ranks, n_grids * (n_vertices ** n_colors))

        self.n_colors = n_colors
        self.n_vertices = n_vertices
        self.n_feats = n_feats
        self.n_ranks = n_ranks
        self.n_grids = n_grids
  
    
    def init_weights(self):
        r"""Init weights for models.

        For the mapping f (`backbone`) and h (`lut_generator`), we follow the initialization in
            [3D-LUT](https://github.com/HuiZeng/Image-Adaptive-3DLUT).

        """
        nn.init.ones_(self.weights_generator.bias)
        nn.init.zeros_(self.basis_grids_bank.bias)
        identity_grid = torch.arange(self.n_vertices).div(self.n_vertices - 1)
        identity_grid = identity_grid.view(1, self.n_vertices, 1, 1)
        identity_grid = identity_grid.repeat(1, 1, self.n_vertices, self.n_vertices)
        identity_grid = torch.cat([identity_grid, torch.zeros(int(self.n_grids/self.n_colors - 1), *((self.n_vertices,) * self.n_colors) )])
        identity_grid = identity_grid.repeat(self.n_colors, 1, 1, 1)
        
        identity_grid = torch.stack([
            identity_grid,
            *[torch.zeros(
                self.n_grids, *((self.n_vertices,) * self.n_colors)) for _ in range(self.n_ranks - 1)]
            ], dim=0).view(self.n_ranks, -1)
        self.basis_grids_bank.weight.data.copy_(identity_grid.t())
        
    def forward(self, img_feature):
        weights = self.weights_generator(img_feature)
        grids = self.basis_grids_bank(weights)
        grids = grids.view([-1,self.n_grids,self.n_vertices,self.n_vertices,self.n_vertices])
        return grids, weights

    
class Slice(nn.Module):
    def __init__(self, grid_interpolation ='tetra', n_inchannel=9):
        super(Slice, self).__init__()
        
        self.conv_1d = nn.Conv2d(n_inchannel + 3, 3, kernel_size=1, padding=0)
        self.n_inchannel = n_inchannel
        
        if grid_interpolation =='tetra':
            self.bilateral_slice = tetrahedral_slice_function
            print('GRID tetrahedral_interpolation apply')
        else:
            self.bilateral_slice = trilinear_slice_function
            print('GRID trilinear_interpolation apply')
    
    
        
    def forward(self, bilateral_grid, guidemap): 
       
        res = self.bilateral_slice(bilateral_grid, guidemap)
        
        res = torch.cat((res, guidemap), dim=1)
        res = self.conv_1d(res)
                
        return res        

class LUTwithBGrid(nn.Module):
    def __init__(self, backbone_type='cnn', backbone_coef=8,
                 lut_n_vertices=17, lut_n_ranks=24, lut_interpolation='tetra', 
                 grid_n_vertices=17, grid_n_ranks=24, grid_interpolation='tri', n_grids=9):
        super(LUTwithBGrid, self).__init__()
        
        self.backbone_type = backbone_type.lower()
        
        if backbone_type.lower() == 'resnet':
            self.backbone = resnet18_224()
            print('Resnet backbone apply')
            n_feats = 512
            #n_feats = 256
        else:
            self.backbone = Backbone(backbone_coef=backbone_coef)
            print('CNN backbone apply')
            n_feats = 32*backbone_coef

        self.gen_3d_lut = Gen_3D_LUT(n_vertices=lut_n_vertices, n_feats=n_feats, n_ranks=lut_n_ranks)
        self.gen_bilateral = Gen_bilateral_grids(n_vertices=grid_n_vertices, n_feats=n_feats, n_ranks=grid_n_ranks, n_grids=n_grids)
        
        self.slice = Slice(grid_interpolation=grid_interpolation, n_inchannel = n_grids)
       
        if lut_interpolation =='tetra':
            self.interpolation = tetrahedral_lut_transform
            print('LUT tetrahedral_lut_transform apply')
        else:
            self.interpolation = trilinear_lut_transform
            print('LUT trilinear_lut_transform apply')
        self.relu = nn.ReLU()
    
    def init_weights(self):
        def special_initilization(m):
            classname = m.__class__.__name__
            if 'Conv' in classname:
                nn.init.xavier_normal_(m.weight.data)
            elif 'InstanceNorm' in classname:
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0.0)
        
        if self.backbone_type != 'resnet':
            self.backbone.apply(special_initilization)
            
        self.gen_3d_lut.init_weights()
        self.gen_bilateral.init_weights()

        self.slice.apply(special_initilization)
        
    def forward(self, img):
        img_feature = self.backbone(img)
        
        g3d_lut, lut_weights = self.gen_3d_lut(img_feature)
        
        gbilateral, grid_weights = self.gen_bilateral(img_feature)
        
        output = self.slice(gbilateral, img)
        output = torch.clamp(output, 0, 1)
        
        output = self.interpolation(g3d_lut, output)
        output = self.relu(output)
        return output, lut_weights, grid_weights, g3d_lut, gbilateral
