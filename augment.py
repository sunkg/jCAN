import time
import numpy as np
import torch
from model import gradient_loss
import torchvision.transforms.functional as F

def rigid_grid(img):
    # rotate and tranlate batch
    rotation = 2*np.pi*0.005
    translation = 0.05
    affines = []
    r_s = np.random.uniform(-rotation, rotation, img.shape[0])
    t_s = np.random.uniform(-translation, translation, img.shape[0])
    for r, t in zip(r_s, t_s):
        # convert origin to center
        # rotation
        R = np.array([ \
                [np.cos(r), -np.sin(r), 0], \
                [np.sin(r),  np.cos(r), 0], \
                [0, 0, 1]])
        # translation
        T = np.array([ \
                [1, 0, t], \
                [0, 1, t], \
                [0, 0, 1]])
        M = T@R # the center is already (0,0), no need to T1, T2
        affines.append(M[:-1])
    M = np.stack(affines, 0)
    M = torch.as_tensor(M, dtype=img.dtype).to(img, non_blocking=True)
    grid = torch.nn.functional.affine_grid(M, \
            size=img.shape, align_corners=False)
    return grid

def bspline_grid(img):
    # rotate and tranlate batch
    scale = 50*2
    grid = (torch.rand(img.shape[0], 2, 9, 9, \
            device=img.device, dtype=img.dtype) - 0.5)/scale
    grid = torch.nn.functional.interpolate(grid, \
            size=img.shape[2:], align_corners=False, mode='bicubic')
    grid = grid.permute(0, 2, 3, 1).contiguous()
    return grid


def rotate(batch):
    W, H = batch[0].shape[-2:]
    for i in range(batch[0].shape[0]):
        for j in range(len(batch)):
            angle = np.random.normal(0,0.25) # mean =0, std = 1.5 degree
            center = [np.random.normal(W//2, W//5), np.random.normal(H//2, H//5)]
            center = np.clip(center, [W//2-W//4,H//2-H//4], [W//2+W//4, H//2+H//4])
            batch_real = F.rotate(
                batch[j][i,...].real, angle=angle, interpolation = 2, center = [center[0],center[1]]
            )
            batch_imag = F.rotate(
                batch[j][i,...].imag, angle=angle, interpolation = 2, center = [center[0],center[1]]
            )
            batch[j][i,...] = torch.complex(batch_real, batch_imag)
    return batch


def flip(batch, num_modal):                
    assert len(batch)%num_modal==0 
    for i in range(batch[0].shape[0]):
        p=torch.rand(1)
        if p>0.5: 
            for j in range(num_modal):
                batch[j][i,...] = F.hflip(batch[j][i,...])
    return batch


def augment(img, rigid=True, bspline=True, grid=None):
    if grid is None:
        assert rigid == True
        img_abs = img.abs()
        grid = rigid_grid(img_abs)
        if bspline:
            grid = grid + bspline_grid(img.abs())
    else:
        assert rigid == False
        assert bspline == False
    sample = lambda x: torch.nn.functional.grid_sample(x, grid, \
            padding_mode='reflection', align_corners=False, mode='bilinear')
    if torch.is_complex(img):
        img = sample(img.real) + sample(img.imag)*1j
    else:
        img = sample(img)

    return img

def augment_eval(img, rigid=True, bspline=True, grid=None):
    if grid is None:
        assert rigid == True
        img_abs = img.abs()
        grid = rigid_grid(img_abs)
        if bspline:
            grid = grid + bspline_grid(img.abs())
    else:
        assert rigid == False
        assert bspline == False
    sample = lambda x: torch.nn.functional.grid_sample(x, grid, \
            padding_mode='reflection', align_corners=False, mode='bilinear')
    if torch.is_complex(img):
        img = sample(img.real) + sample(img.imag)*1j
    else:
        img = sample(img)

    return img, grid


if __name__ == '__main__':
    #img = torch.randn(2, 3, 100, 100)#.cuda()
    img = torch.zeros(10, 1, 321, 321)
    img[:,:,::10] = 0.5
    #img[:,:,5::10] = 1
    img[:,:,:,::10] = 0.5
    #img[:,:,:,5::10] = 1
    img[:,:,[0,160,-1]] = 1
    img[:,:,:,[0,160,-1]] = 1
    #img1, grid = augment(img, bspline=False)
    img1, grid = augment(img)
    same = torch.tensor([[[1,0,0],[0,1,0]]], dtype=img.dtype, device=img.device)
    same = torch.nn.functional.affine_grid(same, \
            size=(1, *img.shape[1:]), align_corners=False)
    img1, img = img1.cpu().numpy(), img.cpu().numpy()
    import matplotlib.pyplot as plt
    #plt.figure()
    #plt.imshow(img[0][0])
    for _img, _img1 in zip(img, img1):
        plt.figure()
        disp = np.concatenate([_img, _img1, _img1], 0)
        plt.imshow(np.moveaxis(disp, 0, -1))
        plt.show()
