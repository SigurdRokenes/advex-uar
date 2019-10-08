"""
Create an augmented dataset
"""

import os
import torch
import numpy as np
from PIL import Image
import argparse

from advex_uar.attacks.snow import snow_creator, make_kernels
from advex_uar.attacks.snow_attack import apply_snow


def create_snow_mask(img, outsize):
    img2 = np.array(img.resize((128,128))).transpose(2,0,1)
    img_t = torch.FloatTensor(img2).unsqueeze(0)

    kernels = make_kernels()
    intensities = torch.exp( (-1/0.2)  * torch.rand(1, 7, outsize//4, outsize//4) )

    snow_mask =  snow_creator(intensities, kernels, resol=outsize)
    img_masked = apply_snow(img_t/255., snow_mask.cpu(), scale=0.2 * torch.ones(1))

    return img_masked[0].detach().cpu().numpy().transpose(1,2,0)

def save_img(img, outdir, bname):
    imgint = (img * 255).astype(np.uint8)
    outpath = os.path.join(outdir, 'snowy_'+bname)
    Image.fromarray(imgint).save(outpath)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--idir', '-i', help='Input directory with images',  required=True)
    parser.add_argument('--odir', '-o', help='Output directory', required=True)
    parser.add_argument('--osize', '-os', help='Size of output image', default=128, type=int)

    args = parser.parse_args()

    for r, d, files in os.walk(args.idir):
        for idx, f in enumerate(files):
            print('Snowing on img {}'.format(idx))
            bname = f
            path = os.path.join(r, f)
            img = Image.open(path)
            snowy_img = create_snow_mask(img, args.osize)
            save_img(snowy_img, args.odir, bname)


