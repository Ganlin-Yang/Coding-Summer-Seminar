import os
import numpy as np
import h5py
import torch
from torchvision import transforms
from PIL import Image


def read_image(filedir):
    files = open(filedir)
    data = []
    return 

def write_image(path,x):
    img = Image.fromarray(np.uint8((x*255).transpose(1, 2, 0)))
    img.save(path)
    return 

def write_h5_geo(filedir, coords):
    data = coords.astype('uint8')
    with h5py.File(filedir, 'w') as h:
        h.create_dataset('data', data=data, shape=data.shape)

    return

def read_ply_ascii_geo(filedir):
    files = open(filedir)
    data = []
    for i, line in enumerate(files):
        wordslist = line.split(' ')
        try:
            line_values = []
            for i, v in enumerate(wordslist):
                if v == '\n': continue
                line_values.append(float(v))
        except ValueError: continue
        data.append(line_values)
    data = np.array(data)
    coords = data[:,0:3].astype('int')

    return coords

def write_ply_ascii_geo(filedir, coords):
    if os.path.exists(filedir): os.system('rm '+filedir)
    f = open(filedir,'a+')
    f.writelines(['ply\n','format ascii 1.0\n'])
    f.write('element vertex '+str(coords.shape[0])+'\n')
    f.writelines(['property float x\n','property float y\n','property float z\n'])
    f.write('end_header\n')
    coords = coords.astype('int')
    for p in coords:
        f.writelines([str(p[0]), ' ', str(p[1]), ' ',str(p[2]), '\n'])
    f.close() 

    return

def array2vector(array, step):
    """ravel 2D array with multi-channel to one 1D vector by sum each channel with different step.
    """
    array, step = array.long().cpu(), step.long().cpu() 
    vector = sum([array[:,i]*(step**i) for i in range(array.shape[-1])])

    return vector

def crop(x):
    # Input: x Tensor判断x的H，W是否为64的整数倍，若是直接返回x，若不是则利用中心裁剪
    H, W = x.shape[-2:]
    if H%64==0 and W%64==0:
        output = x
    else:
        H = H-H%64
        W = W-W%64
        crop_transforms = transforms.Compose(
        [transforms.CenterCrop((H, W))]
        )
        output = crop_transforms(x)
    return output

def cal_psnr(x, x_dec):
    # Input: x, x_dec: Tensor[B,C,H,W]
    PIXEL_MAX = 255
    x = x.cpu().numpy().astype(float)*255
    x_dec = x_dec.cpu().numpy().astype(float)*255.
    mse = np.mean((x-x_dec)**2)
    if mse == 0:
        return 100
    return 20*np.log10(PIXEL_MAX/np.sqrt(mse))
