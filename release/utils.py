# -*- coding: utf-8 -*-
"""
@author: sunkg
"""
import torch.nn as nn
from timm.models.layers import to_2tuple
import math
import torch
import numpy as np
import torch.fft
import torch.nn.functional as F
from unet import UNet
import nibabel as nib
import h5py

def fft2(x):
    assert len(x.shape) == 4
    x = torch.fft.fft2(x, norm='ortho')
    return x

def ifft2(x):
    assert len(x.shape) == 4
    x = torch.fft.ifft2(x, norm='ortho')
    return x

def fftshift2(x):
    assert len(x.shape) == 4
    x = torch.roll(x, (x.shape[-2]//2, x.shape[-1]//2), dims=(-2, -1))
    return x

def ifftshift2(x):
    assert len(x.shape) == 4
    x = torch.roll(x, ((x.shape[-2]+1)//2, (x.shape[-1]+1)//2), dims=(-2, -1))
    return x

def rss(x):
    assert len(x.shape) == 4
    return torch.linalg.vector_norm(x, ord=2, dim=1, keepdim=True)

def rss2d(x):
    assert len(x.shape) == 2
    return (x.real**2 + x.imag**2).sqrt()
    
def ssimloss(X, Y):
    assert not torch.is_complex(X)
    assert not torch.is_complex(Y)
    win_size = 7
    k1 = 0.01
    k2 = 0.03
    w = torch.ones(1, 1, win_size, win_size).to(X) / win_size ** 2
    NP = win_size ** 2
    cov_norm = NP / (NP - 1)
    data_range = 1
    C1 = (k1 * data_range) ** 2
    C2 = (k2 * data_range) ** 2
    ux = F.conv2d(X, w)
    uy = F.conv2d(Y, w)
    uxx = F.conv2d(X * X, w)
    uyy = F.conv2d(Y * Y, w)
    uxy = F.conv2d(X * Y, w)
    vx = cov_norm * (uxx - ux * ux)
    vy = cov_norm * (uyy - uy * uy)
    vxy = cov_norm * (uxy - ux * uy)
    A1, A2, B1, B2 = (
        2 * ux * uy + C1,
        2 * vxy + C2,
        ux ** 2 + uy ** 2 + C1,
        vx + vy + C2,
    )
    D = B1 * B2
    S = (A1 * A2) / D
    return 1 - S.mean()

def gaussian_kernel_1d(sigma):
    kernel_size = int(2*math.ceil(sigma*2) + 1)
    x = torch.linspace(-(kernel_size-1)//2, (kernel_size-1)//2, kernel_size).cuda()
    kernel = 1.0/(sigma*math.sqrt(2*math.pi))*torch.exp(-(x**2)/(2*sigma**2))
    kernel = kernel/torch.sum(kernel)
    return kernel

def gaussian_kernel_2d(sigma):
    y_1 = gaussian_kernel_1d(sigma[0])
    y_2 = gaussian_kernel_1d(sigma[1])
    kernel = torch.tensordot(y_1, y_2, 0)
    kernel = kernel / torch.sum(kernel)
    return kernel

def gaussian_smooth(img, sigma):
    sigma = max(sigma, 1e-12)
    kernel = gaussian_kernel_2d((sigma, sigma))[None, None, :, :].to(img)
    padding = kernel.shape[-1]//2
    img = torch.nn.functional.conv2d(img, kernel, padding=padding)
    return img

def compute_marginal_entropy(values, bins, sigma):
    normalizer_1d = np.sqrt(2.0 * np.pi) * sigma
    sigma = 2*sigma**2
    p = torch.exp(-((values - bins).pow(2).div(sigma))).div(normalizer_1d)
    p_n = p.mean(dim=1)
    p_n = p_n/(torch.sum(p_n) + 1e-10)
    return -(p_n * torch.log(p_n + 1e-10)).sum(), p

def _mi_loss(I, J, bins, sigma):
    # compute marjinal entropy
    ent_I, p_I = compute_marginal_entropy(I.view(-1), bins, sigma)
    ent_J, p_J = compute_marginal_entropy(J.view(-1), bins, sigma)
    # compute joint entropy
    normalizer_2d = 2.0 * np.pi*sigma**2
    p_joint = torch.mm(p_I, p_J.transpose(0, 1)).div(normalizer_2d)
    p_joint = p_joint / (torch.sum(p_joint) + 1e-10)
    ent_joint = -(p_joint * torch.log(p_joint + 1e-10)).sum()

    return -(ent_I + ent_J - ent_joint)

def mi_loss(I, J, bins=64 ,sigma=1.0/64, minVal=0, maxVal=1):
    bins = torch.linspace(minVal, maxVal, bins).to(I).unsqueeze(1)
    neg_mi =[_mi_loss(I, J, bins, sigma) for I, J in zip(I, J)]
    return sum(neg_mi)/len(neg_mi)

def ms_mi_loss(I, J, bins=64, sigma=1.0/64, ms=3, smooth=3, minVal=0, maxVal=1):
    smooth_fn = lambda x: torch.nn.functional.avg_pool2d( \
            gaussian_smooth(x, smooth), kernel_size = 2, stride=2)
    loss = mi_loss(I, J, bins=bins, sigma=sigma, minVal=minVal, maxVal=maxVal)
    for _ in range(ms - 1):
        I, J = map(smooth_fn, (I, J))
        loss = loss + mi_loss(I, J, \
                bins=bins, sigma=sigma, minVal=minVal, maxVal=maxVal)
    return loss / ms

def correlation_loss(SM):
    B, C, H, W = SM.shape
    SM_ = SM.view(B, C, -1)
    loss = 0
    for i in range(B):
        cc = torch.corrcoef(SM_[i, ...])
        loss += F.l1_loss(cc, torch.eye(C).cuda())
    return loss

def gradient(x,h_x=None,w_x=None):
    if h_x is None and w_x is None:
        h_x = x.size()[2]
        w_x = x.size()[3]
    r = F.pad(x, (0, 1, 0, 0))[:, :, :, 1:]
    l = F.pad(x, (1, 0, 0, 0))[:, :, :, :w_x]
    t = F.pad(x, (0, 0, 1, 0))[:, :, :h_x, :]
    b = F.pad(x, (0, 0, 0, 1))[:, :, 1:, :]
    xgrad = torch.pow(torch.pow((r - l), 2) + torch.pow((t - b), 2), 0.5)
    return xgrad


def convert(nii_path, h5_path, protocal):
    # convert nii file with path nii_path to h5 file stored at h5_path
    # protocal name as string
    h5 = h5py.File(h5_path, 'w')
    nii = nib.load(nii_path)
    array = nib.as_closest_canonical(nii).get_fdata() #convert to RAS
    array = array.T.astype(np.float32)
    h5.create_dataset('image', data=array)
    h5.attrs['max'] = array.max()
    h5.attrs['acquisition'] = protocal
    h5.close()
    

def crop_Kspace(Kspace_f, SR_scale):
    Kspace_f = fftshift2(Kspace_f)
    B, C, H, W = Kspace_f.shape
    if SR_scale!=1:
        margin_H = int(H*(SR_scale-1)/SR_scale//2)
        margin_W = int(W*(SR_scale-1)/SR_scale//2)
        Kspace_f = Kspace_f[:,:, margin_H:margin_H+H//SR_scale, margin_W:margin_W+W//SR_scale] # roll it by half
    Kspace_f = ifftshift2(Kspace_f)
    return Kspace_f


def complex_to_chan_dim(x: torch.Tensor) -> torch.Tensor:
    assert torch.is_complex(x)
    return torch.cat([x.real, x.imag], dim=1)


def chan_dim_to_complex(x: torch.Tensor) -> torch.Tensor:
    assert not torch.is_complex(x)
    _, c, _, _ = x.shape
    assert c % 2 == 0
    c = c // 2
    return torch.complex(x[:,:c], x[:,c:])


def UpImgComplex(img_complex, SR_scale):
    img_real=complex_to_chan_dim(img_complex)
    img_real=nn.functional.interpolate(img_real, scale_factor=SR_scale, mode='bicubic')
    return chan_dim_to_complex(img_real)


def norm(x: torch.Tensor):
    # group norm
    b, c, h, w = x.shape
    assert c%2 == 0
    x = x.view(b, 2, c // 2 * h * w)

    mean = x.mean(dim=2).view(b, 2, 1)
    std = x.std(dim=2).view(b, 2, 1)

    x = (x - mean) / (std + 1e-12)

    return x.view(b, c, h, w), mean, std
    
def unnorm(
    x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
    ) :
    b, c, h, w = x.shape
    assert c%2 == 0
    x = x.view(b, 2, c // 2 * h * w)
    x = x* std + mean
    return x.view(b, c, h, w)
        
def preprocess(x):
    assert torch.is_complex(x)
    x = complex_to_chan_dim(x)
    x, mean, std = norm(x)
    return x, mean, std

def postprocess(x, mean, std):
    x = unnorm(x, mean, std)
    x = chan_dim_to_complex(x)
    return x
  
def pad(x, window_size):
    _, _, h, w = x.size()
    mod_pad_h = (window_size - h % window_size) % window_size
    mod_pad_w = (window_size - w % window_size) % window_size
    x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
    return x, (mod_pad_w, mod_pad_h)

def unpad(
    x: torch.Tensor,
    w_pad: int,
    h_pad: int
) -> torch.Tensor:
    return x[...,0 : x.shape[-2] - h_pad, 0 : x.shape[-1] - w_pad]
    
def check_image_size(img_size, window_size):
    h, w = img_size
    mod_pad_h = (window_size - h % window_size) % window_size
    mod_pad_w = (window_size - w % window_size) % window_size
    return h + mod_pad_h, w + mod_pad_w
    
def sens_expand(image: torch.Tensor, sens_maps: torch.Tensor) -> torch.Tensor:
    return fft2(image * sens_maps)

def sens_reduce(kspace: torch.Tensor, sens_maps: torch.Tensor) -> torch.Tensor:
    return (ifft2(kspace) * sens_maps.conj()).sum(dim=1, keepdim=True)
    
class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 2.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size, patch_size=4, in_chans=2, embed_dim=96, norm_layer=None):
        super().__init__()
        self.img_size = img_size
        self.patch_size = to_2tuple(patch_size)
        self.patches_resolution = tuple([self.img_size[0] // patch_size, self.img_size[1] // patch_size])
        self.num_patches = self.patches_resolution[0] * self.patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):        
        x = self.proj(x).flatten(2).transpose(1, 2) # B N_patch C	
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        flops = 0
        H, W = self.img_size
        if self.norm is not None:
            flops += H * W * self.embed_dim
        return flops


class PatchUnEmbed(nn.Module):
    r""" Image to Patch Unembedding

    Args:
        img_size (int): Image size. 
        patch_size (int): Patch token size. Default: 4.
        out_chans (int): Number of output image channels. Default: 2.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size, patch_size=4, out_chans=2, embed_dim=96, norm_layer=None):
        super().__init__()
        self.img_size = img_size  
        self.patch_size = patch_size
        self.patches_resolution = tuple([self.img_size[0] // patch_size, self.img_size[1] // patch_size])
        self.out_chans = out_chans
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(embed_dim, out_chans, kernel_size=int(np.floor(patch_size*1.5)), stride=1, padding='same')
        #### with gated conv ####
        self.gate_proj = nn.Conv2d(embed_dim, out_chans, kernel_size=int(np.floor(patch_size*1.5)), stride=1, padding='same')
        self.act_layer = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.scale = nn.Parameter(torch.rand(1))

    def forward(self, x):
        x = x.transpose(1, 2).view(-1, self.embed_dim, self.patches_resolution[0], self.patches_resolution[1])
        x = nn.functional.interpolate(x, scale_factor=self.patch_size, mode='bicubic')
        #### with gated conv ####
        x_origin = self.act_layer(self.proj(x))
        x_gate = torch.sigmoid(self.scale*self.gate_proj(x))
        x = x_origin * x_gate
        return x

    def flops(self):
        flops = 0
        return flops


class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        return out
    
    
class Upsample(nn.Sequential):
    """Upsample module.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """

    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported. ' 'Supported scales: 2^n and 3.')
        super(Upsample, self).__init__(*m)


class UpsampleOneStep(nn.Sequential):
    """UpsampleOneStep module (the difference with Upsample is that it always only has 1conv + 1pixelshuffle)
       Used in lightweight SR to save parameters.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.

    """

    def __init__(self, scale, num_feat, num_out_ch, input_resolution=None):
        self.num_feat = num_feat
        self.input_resolution = input_resolution
        m = []
        m.append(nn.Conv2d(num_feat, (scale ** 2) * num_out_ch, 3, 1, 1))
        m.append(nn.PixelShuffle(scale))
        super(UpsampleOneStep, self).__init__(*m)

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.num_feat * 3 * 9
        return flops


class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.dim
        flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        return flops
								
								
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x
