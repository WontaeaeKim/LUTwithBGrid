import argparse

import torch

import torchvision.transforms.functional as TF
import torchvision.transforms as T

from models import *
from PIL import Image
import cv2



parser = argparse.ArgumentParser()
parser.add_argument("--input_path", type=str, default="./demo_img/input/a0634.jpg", help="path of pretrained model")
parser.add_argument("--output_path", type=str, default="./demo_img/result/a0634.png", help="path of pretrained model")
parser.add_argument("--dataset_name", type=str, default="fivek", help="name of the dataset: fivek or ppr10k")
parser.add_argument("--input_color_space", type=str, default="sRGB", help="input color space: sRGB or XYZ")
parser.add_argument("--lut_inter", type=str, default="tri", help="LUT interpolation method")
parser.add_argument("--pretrained_path", type=str, default="./pretrained/FiveK_sRGB.pth", help="path of pretrained model")

parser.add_argument("--lut_n_vertices", type=int, default="17", help="number of LUT vertices")
#parser.add_argument("--lut_n_ranks", type=int, default="10", help="number of LUT generator ranks")
parser.add_argument("--lut_interpolation", type=str, default="tetra", help="method of LUT grid interpolation")
    
parser.add_argument("--grid_n_vertices", type=int, default="17", help="number of GRID vertices")
parser.add_argument("--grid_n_ranks", type=int, default="8", help="number of GRID generator ranks")
parser.add_argument("--grid_interpolation", type=str, default="tri", help="method of GRID grid interpolation")
parser.add_argument("--n_grids", type=int, default="6", help="number of GRID generator output channel")

opt = parser.parse_args()

# use gpu when detect cuda
cuda = True if torch.cuda.is_available() else False

# Tensor type
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

criterion_pixelwise = torch.nn.MSELoss()

if opt.dataset_name == "ppr10k":
    backbone_type = 'resnet'
    lut_n_ranks = 10
else:
    backbone_type = 'cnn'
    lut_n_ranks = 8    


lut_bgrid_inst = LUTwithBGrid(backbone_type=backbone_type, 
                 lut_n_vertices=opt.lut_n_vertices, lut_n_ranks=lut_n_ranks, lut_interpolation=opt.lut_interpolation, 
                 grid_n_vertices=opt.grid_n_vertices, grid_n_ranks=opt.grid_n_ranks, grid_interpolation=opt.grid_interpolation, n_grids=opt.n_grids)


if cuda:
    lut_bgrid_inst = lut_bgrid_inst.cuda()
    criterion_pixelwise.cuda()

# Load pretrained models
lut_bgrid_inst.load_state_dict(torch.load(opt.pretrained_path))
lut_bgrid_inst.eval()



img = Image.open(opt.input_path)
real_A = TF.to_tensor(img).type(Tensor)
real_A = real_A.unsqueeze(0)
result, _, _, _, _ = lut_bgrid_inst(real_A)

result = result.squeeze().mul_(255).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
cv2.imwrite(opt.output_path, result)


