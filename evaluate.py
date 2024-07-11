import argparse
import time
import torch
import math
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.autograd import Variable

from models import *
from datasets import *



parser = argparse.ArgumentParser()
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

if opt.input_color_space == 'XYZ':
    dataloader = DataLoader(
        ImageDataset_XYZ("../../dataset/%s" % "fiveK",  mode="test"),
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )
else:
    if opt.dataset_name == "ppr10k":
        dataloader = DataLoader(
            ImageDataset_PPR10k("../../dataset/%s" % "ppr10k",  mode="test"),
            batch_size=1,
            shuffle=False,
            num_workers=0,
        )
    else:
        dataloader = DataLoader(
            ImageDataset_sRGB("../../dataset/%s" % "fiveK",  mode="test"),
            batch_size=1,
            shuffle=False,
            num_workers=0,
        )


def visualize_result():
    """Saves a generated sample from the validation set"""
    out_dir = "images/%s_%s" % (opt.dataset_name, opt.input_color_space)
    os.makedirs(out_dir, exist_ok=True)
    for i, batch in enumerate(dataloader):
        real_A = Variable(batch["A_input"].type(Tensor))
        img_name = batch["input_name"]
        fake_B, _, _, _, _ = lut_bgrid_inst(real_A)

        save_image(fake_B, os.path.join(out_dir,"%s.png" % (img_name[0][:-4])), nrow=1, normalize=False)


def test_speed():
    for i in range(1,10):
        img_input = Image.open(os.path.join("../../../dataset/fiveK/input/JPG","480p","a000%d.jpg"%i))
        img_input = torch.unsqueeze(TF.to_tensor(TF.resize(img_input,(2160,3840))),0)
        real_A = Variable(img_input.type(Tensor))
        torch.cuda.synchronize()
        t0 = time.time()
        for j in range(0,100):
            fake_B, _, _, _, _ = lut_bgrid_inst(real_A)
        
        torch.cuda.synchronize()
        t1 = time.time()
        print((t1 - t0))

# ----------
#  evaluation
# ----------
visualize_result()
#test_speed()


