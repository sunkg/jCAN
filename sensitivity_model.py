#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 07:21:26 2022

@author: sunkg
""" 

import torch.nn as nn
import torch
import utils
import torch.nn.functional as F
import math
from typing import List, Tuple, Optional
from CNN import ConvBlock

class SensitivityModel(nn.Module):
    """
    Model for learning sensitivity estimation from k-space data.

    This model applies an IFFT to multichannel k-space data and then a U-Net
    to the coil images to estimate coil sensitivities. It can be used with the
    end-to-end variational network.

    Note SensitivityModel is designed for complex input/output only.
    """

    def __init__(
        self,
        chans: int,
        sens_steps: int,
        mask_center: bool = True
    ):
        """
        Args:
            chans: Number of output channels of the first convolution layer.
            num_pools: Number of down-sampling and up-sampling layers.
            in_chans: Number of channels in the complex input.
            out_chans: Number of channels in the complex output.
            mask_center: Whether to mask center of k-space for sensitivity map
                calculation.
        """
        super().__init__()
        self.mask_center = mask_center

        self.norm_net = NormNet(
            chans,
            sens_steps
            )

    '''
    def up(self, x):
        xR, xI = x.real, x.imag
        xR = F.interpolate(xR, scale_factor=2, mode='bilinear')
        xI = F.interpolate(xI, scale_factor=2, mode='bilinear')
        return torch.complex(xR, xI)

    def down(self, x):
        xR, xI = x.real, x.imag
        xR = F.avg_pool2d(xR, 2)
        xI = F.avg_pool2d(xI, 2)
        return torch.complex(xR, xI)
    '''

    def forward(
        self,
        masked_kspace: torch.Tensor,
        num_low_frequencies: int,
    ) -> torch.Tensor:
        # get ACS signals only (i.e. preserve low freq only)
        ACS_mask = torch.ones(masked_kspace.shape[-1])
        ACS_mask[num_low_frequencies:] = 0
        ACS_mask = torch.roll(ACS_mask, -num_low_frequencies//2)
        ACS_mask = ACS_mask[None, None, None, :].to(masked_kspace)
        ACS_kspace = ACS_mask * masked_kspace

        # convert to image space
        ACS_images = utils.ifft2(ACS_kspace)

        # estimate sensitivities independently
        N, C, H, W = ACS_images.shape
        batched_channels = ACS_images.reshape(N*C, 1, H, W)
        
        #### gated ####
        sensitivity, gate = self.norm_net(batched_channels)
        gate = gate.reshape(N, C, H, W)
        
        sensitivity = sensitivity.reshape(N, C, H, W)
        sensitivity = sensitivity / (utils.rss(sensitivity) + 1e-12)
        
        #### gated ####
        sensitivity = gate * sensitivity     
        
        return sensitivity, gate
    
    
class NormNet(nn.Module):
    """
    Normalized Net model: in Unet or ResNet

    This is the same as a regular U-Net, but with normalization applied to the
    input before the U-Net. This keeps the values more numerically stable
    during training.

    Note NormUnet is designed for complex input/output only.
    """

    def __init__(
        self,
        chans: int,
        num_steps: int
    ):
        """
        Args:
            chans: Number of output channels of the first convolution layer.
            num_pools: Number of down-sampling and up-sampling layers.
            in_chans: Number of channels in the complex input.
            out_chans: Number of channels in the complex output.
        """
        super().__init__()
        
        self.NormNet = Unet(
            chans=chans,
            num_pool_layers=num_steps,
        )

    def complex_to_chan_dim(self, x: torch.Tensor) -> torch.Tensor:
        assert torch.is_complex(x)
        return torch.cat([x.real, x.imag], dim=1)

    def chan_dim_to_complex(self, x: torch.Tensor) -> torch.Tensor:
        assert not torch.is_complex(x)
        _, c, _, _ = x.shape
        assert c % 2 == 0
        c = c // 2
        return torch.complex(x[:,:c], x[:,c:])

    def norm(self, x: torch.Tensor):
        # group norm
        b, c, h, w = x.shape
        assert c%2 == 0
        x = x.view(b, 2, c // 2 * h * w)
    
        mean = x.mean(dim=2).view(b, 2, 1)
        std = x.std(dim=2).view(b, 2, 1)
    
        x = (x - mean) / (std + 1e-12)
    
        return x.view(b, c, h, w), mean, std
        
    def unnorm(self, x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) :
        b, c, h, w = x.shape
        assert c%2 == 0
        x = x.contiguous().view(b, 2, c // 2 * h * w)
        x = x* std + mean
        return x.view(b, c, h, w)

    def pad(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[List[int], List[int], int, int]]:
        _, _, h, w = x.shape
        w_mult = ((w - 1) | 15) + 1
        h_mult = ((h - 1) | 15) + 1
        w_pad = [math.floor((w_mult - w) / 2), math.ceil((w_mult - w) / 2)]
        h_pad = [math.floor((h_mult - h) / 2), math.ceil((h_mult - h) / 2)]
        x = F.pad(x, w_pad + h_pad)

        return x, (h_pad, w_pad, h_mult, w_mult)

    def unpad(
        self,
        x: torch.Tensor,
        h_pad: List[int],
        w_pad: List[int],
        h_mult: int,
        w_mult: int,
    ) -> torch.Tensor:
        return x[..., h_pad[0] : h_mult - h_pad[1], w_pad[0] : w_mult - w_pad[1]]

    def forward(
        self, 
        x: torch.Tensor,
        ref: Optional[torch.Tensor]=None,
    ) -> torch.Tensor:
        assert len(x.shape) == 4
        assert torch.is_complex(x)

        # get shapes for unet and normalize
        x = self.complex_to_chan_dim(x)
        x, mean, std = self.norm(x)
        x, pad_sizes = self.pad(x)
            
        #### gated ####
        x, gate = self.NormNet(x)
        gate = self.unpad(gate, *pad_sizes)
        
        # get shapes back and unnormalize
        x = self.unpad(x, *pad_sizes)
        x = self.unnorm(x, mean, std)
        x = self.chan_dim_to_complex(x)

        return x, gate
   
class ResNet(nn.Module):
    """
    A Residual network of CNN layers.
    """

    def __init__(self, chans_max, num_conv):
        """
        Args:
	        chans_max: Number of maximum channels 
            num_conv: Number of body convs 
        """
        super().__init__()
        self.chans_in = 2
        self.chans_out = 2
        self.chans_max = chans_max
        self.num_conv = num_conv

        self.input_layers = nn.Sequential(
            nn.Conv2d(self.chans_in, self.chans_max, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(self.chans_max),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        
        self.inter_layers = nn.Sequential(
            nn.Conv2d(self.chans_max, self.chans_max, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(self.chans_max),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
								
        self.output_layers = nn.Sequential(
            nn.Conv2d(self.chans_max, self.chans_out, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(self.chans_max),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        
        # define head module
        m_head = [self.input_layers]

        # define body module
        m_body = [self.inter_layers for _ in range(self.num_conv)]

        m_tail = [self.output_layers]
								
        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)
    
    def forward(self, x):
        x_ = self.head(x)
        x_ = self.body(x_)
        res = self.tail(x_)
        output = x + res

        return output 
    

class Unet(nn.Module):
    """
    PyTorch implementation of a U-Net model.

    O. Ronneberger, P. Fischer, and Thomas Brox. U-net: Convolutional networks
    for biomedical image segmentation. In International Conference on Medical
    image computing and computer-assisted intervention, pages 234–241.
    Springer, 2015.
    """

    def __init__(
        self,
        chans: int = 32,
        num_pool_layers: int = 4,
    ):
        """
        Args:
            chans: Number of output channels of the first convolution layer.
            num_pool_layers: Number of down-sampling and up-sampling layers.
        """
        super().__init__()

        self.in_chans = 2
        self.out_chans = 2
        self.chans = chans
        self.num_pool_layers = num_pool_layers

        self.down_sample_layers = nn.ModuleList([ConvBlock(self.in_chans, chans)])
        ch = chans
        for _ in range(num_pool_layers - 1):
            self.down_sample_layers.append(ConvBlock(ch, ch * 2))
            ch *= 2
        self.conv = ConvBlock(ch, ch * 2)

        self.up_conv = nn.ModuleList()
        self.upsampling_conv = nn.ModuleList()

        for _ in range(num_pool_layers - 1):
            self.upsampling_conv.append(ConvBlock(ch * 2, ch))
            self.up_conv.append(ConvBlock(ch * 2, ch)) 
            ch //= 2
            
        self.conv_sl = ConvBlock(ch*2, ch)
        self.norm_conv = nn.Conv2d(ch, self.out_chans, kernel_size=1, stride=1)

        #### gated conv ####    
        self.scale = nn.Parameter(torch.ones(1))
        self.shift = nn.Parameter(torch.zeros(1))
        self.gate_conv = nn.Conv2d(ch, 1, kernel_size=1, stride=1)

        #### learnable Gaussian std ####
        self.stds = nn.ParameterList([nn.Parameter(torch.ones(1)) for _ in range(num_pool_layers)])


    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Input 4D tensor of shape `(N, in_chans, H, W)`.

        Returns:
            Output tensor of shape `(N, out_chans, H, W)`.
        """
        assert not torch.is_complex(image)
        stack = []
        output = image

        # apply down-sampling layers
        for layer in self.down_sample_layers:
            output = layer(output)
            stack.append(output)
            output = F.avg_pool2d(output, kernel_size=2, stride=2, padding=0)

        output = self.conv(output)
        cnt = 0
        # apply up-sampling layers
        for up_conv, conv in zip(self.upsampling_conv, self.up_conv):
 
            #### learnable Gaussian std ####
            output_ = utils.gaussian_smooth(output.view(-1, 1, *output.shape[-2:]), sigma = torch.clamp(self.stds[cnt], min=1e-9, max=7))
            output = output_.view(output.shape)
            
            downsample_layer = stack.pop()
            output = up_conv(output)
            output = F.interpolate(output, scale_factor=2)

            # reflect pad on the right/botton if needed to handle odd input dimensions
            padding = [0, 0, 0, 0]
            if output.shape[-1] != downsample_layer.shape[-1]:
                padding[1] = 1  # padding right
            if output.shape[-2] != downsample_layer.shape[-2]:
                padding[3] = 1  # padding bottom
            if torch.sum(torch.tensor(padding)) != 0:
                output = F.pad(output, padding, "reflect")

            output = torch.cat([output, downsample_layer], dim=1)
            output = conv(output)
            cnt = cnt + 1

        output = self.conv_sl(output)
        #### learnable Gaussian std ####
        output_ = utils.gaussian_smooth(output.view(-1, 1, *output.shape[-2:]), sigma = torch.clamp(self.stds[cnt], min=1e-9, max=7))
        output = output_.view(output.shape)        
        output = F.interpolate(output, scale_factor=2**(self.num_pool_layers-3))       
        norm_conv = self.norm_conv(output)

        #### gated ####
        #gate_conv = torch.sigmoid(1*self.gate_conv(output))
        gate_conv = torch.sigmoid(self.scale*(self.gate_conv(output)+self.shift))
       
        return norm_conv, gate_conv
    
