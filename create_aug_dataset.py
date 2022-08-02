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
from advex_uar.attacks.fog_attack import apply_fog
from advex_uar.attacks.jpeg_attack import apply_jpeg
from advex_uar.attacks.gabor_attack import apply_gabor

def convert_image(img, outsize, none = False):
    if none == True:
        img2 = np.array(img.resize((outsize,outsize)))#.transpose(2,0,1)
        return img2
    else:
        img2 = np.array(img.resize((outsize,outsize))).transpose(2,0,1)
        img_t = torch.FloatTensor(img2).unsqueeze(0).cuda()
        return img_t

def apply_gabor_mask(img_t, outsize, eps_max=6.25):
    img_masked = apply_gabor(img_t / 255, outsize, eps_max)

    #print("not implemented")
    return (img_masked[0].detach().cpu().numpy().transpose(1,2,0) * 255).astype(np.uint8)


def apply_jpeg_mask(img_t, outsize):
    img_masked = apply_jpeg(img_t, outsize, eps_max=0.0625)
    return (img_masked[0].detach().cpu().numpy().transpose(1,2,0) * 255).astype(np.uint8)

def apply_fog_mask(img_t, outsize, eps_max=512):

    #print(img_t)
    img_masked = apply_fog(img_t, outsize, wibble_decay = 1.75, eps_max=eps_max)
    #print(img_masked.shape)
    return (img_masked[0].detach().cpu().numpy().transpose(1,2,0) * 255).astype(np.uint8)

def create_snow_mask(img_t, outsize, eps_max = 0.2):
    scale = eps_max
    kernels = make_kernels()
    intensities = torch.exp( (-1/scale)  * torch.rand(1, 7, outsize//4, outsize//4) )

    snow_mask =  snow_creator(intensities, kernels, resol=outsize)
    img_masked = apply_snow(img_t/255., snow_mask.cpu(), scale=scale * torch.ones(1))

    return (img_masked[0].detach().cpu().numpy().transpose(1,2,0) * 255).astype(np.uint8)


def save_img(img, outdir, bname, dist_type, eps_max):
    imgint = (img).astype(np.uint8)
    folder = os.path.join(outdir, dist_type, str(eps_max))
    #print(folder)
    if not os.path.exists(folder):
        #print("this not happens")
        os.makedirs(folder)
    outpath = os.path.join(folder, '{}_'.format(dist_type)+bname)

    Image.fromarray(imgint).save(outpath)


def main():


    parser = argparse.ArgumentParser()
    parser.add_argument('--idir', '-i', help='Input directory with images', default="images", type=str) #required=True)
    parser.add_argument('--odir', '-o', help='Output directory', default="output_images", type=str)#required=True)
    parser.add_argument('--osize', '-os', help='Size of output image', default=1024, type=int)
    parser.add_argument('--type', '-t', help= "Type of distortion (jpeg, fog, snow, gabor, none", default='gabor', type=str)
    parser.add_argument('--eps', '-e', help= "Distortion strength", default=6.25, type=float)
    args = parser.parse_args()

    for r, d, files in os.walk(args.idir):
        for idx, f in enumerate(files):
            

            bname = f
            path = os.path.join(r, f)
            eps_max = args.eps
            im = Image.open(path)
            img = convert_image(im, outsize=args.osize)

            #print(torch.max(img))
            if args.type == 'snow':
                ### Comment out this and uncomment block below. Own testing
                for val in [0.0625, 0.125, 0.25]:

                    print('Snowing on img {}'.format(idx))
                    snowy_img = create_snow_mask(img, args.osize, val)
                    save_img(snowy_img, args.odir, bname, 'snowy', val)
                """
                print('Snowing on img {}'.format(idx))
                snowy_img = create_snow_mask(img, args.osize, eps_max)
                save_img(snowy_img, args.odir, bname, 'snowy', eps_max)
                """
            
            if args.type == 'fog':
                for val in [128.0, 256.0, 512.0]:
                    print('Fog on img {}'.format(idx))
                    foggy_img = apply_fog_mask(img, args.osize, val)
                    save_img(foggy_img, args.odir, bname, 'foggy', val)
                    #print('Not implemented')
            if args.type == 'jpeg':
                #print('JPEG applied to img {}'.format(idx))
                #jpeg_img = apply_jpeg_mask(img, args.osize, eps_max)
                #save_img(jpeg_img, args.odir, bname, 'jpeg', eps_max)
                print('Not implemented')
                break
            if args.type == 'gabor':
                for val in [6.25, 25.0, 12.5]:
                    print('Gabor applied to img {}'.format(idx))
                    gabor_img = apply_gabor_mask(img, args.osize, val)
                    save_img(gabor_img, args.odir, bname, 'gabor', val)
            if args.type == 'none':
                img = convert_image(im, outsize=args.osize, none = True)
                folder = os.path.join(args.odir, 'none')
                #print(folder)
                if not os.path.exists(folder):
                    #print("this not happens")
                    os.makedirs(folder)
                outpath = os.path.join(folder, '{}_'.format('none')+bname)

                Image.fromarray(img).save(outpath)
                #save_img(img, args.odir, bname, 'none', 0)
                #print('Not implemented')

if __name__ == '__main__':
    main()