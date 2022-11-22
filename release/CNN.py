"""
@author: sunkg
"""

import torch
from torch import nn
from torch.nn import functional as F
import utils
import numpy as np

class ResNet(nn.Module):
    """
    A convolutional network with residual connections.
    """

    def __init__(self, chans_in, chans_max, num_conv, channel_scale=2):
        """
        Args:
            chans_in: Number of channels in the input.  
	       chans_max: Number of maximum channels 
            num_conv: Number of convs in the body.
            channel_scale: combine two modalities in channel.
        """
        super().__init__()
        self.chans_in = chans_in
        self.chans_out = chans_in//channel_scale
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
        m_body = [
             self.inter_layers for _ in range(self.num_conv)
        ]

        m_tail = [self.output_layers]
								
        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)
    
    def forward(self, x):
        x_ = self.head(x)
        x_ = self.body(x_)
        res = self.tail(x_)
        output = x[:,self.chans_in//2:,:,:] + res

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
        in_chans: int,
        out_chans: int,
        chans: int = 32, 
        num_pool_layers: int = 3,
    ):
        """
        Args:
            in_chans: Number of channels in the input to the U-Net model.
            out_chans: Number of channels in the output to the U-Net model.
            chans: Number of output channels of the first convolution layer.
            num_pool_layers: Number of down-sampling and up-sampling layers.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans = chans
        self.num_pool_layers = num_pool_layers

        self.down_sample_layers = nn.ModuleList([ConvBlock(in_chans, chans)])
        ch = chans
        for _ in range(num_pool_layers - 1):
            self.down_sample_layers.append(ConvBlock(ch, ch * 2))
            ch *= 2
        self.conv = ConvBlock(ch, ch * 2)

        self.up_conv = nn.ModuleList()
        self.up_transpose_conv = nn.ModuleList()
        for _ in range(num_pool_layers - 1):
            self.up_transpose_conv.append(TransposeConvBlock(ch * 2, ch))
            self.up_conv.append(ConvBlock(ch * 2, ch))
            ch //= 2

        self.up_transpose_conv.append(TransposeConvBlock(ch * 2, ch))
        self.up_conv.append(
            nn.Sequential(
                ConvBlock(ch * 2, ch),
                nn.Conv2d(ch, self.out_chans, kernel_size=1, stride=1),
            )
        )

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

        # apply up-sampling layers
        for transpose_conv, conv in zip(self.up_transpose_conv, self.up_conv):
            downsample_layer = stack.pop()
            output = transpose_conv(output)

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

        return output


class ConvBlock(nn.Module):
    """
    A Convolutional Block that consists of two convolution layers each followed by
    instance normalization, LeakyReLU activation and dropout.
    """

    def __init__(self, in_chans: int, out_chans: int):
        """
        Args:
            in_chans: Number of channels in the input.
            out_chans: Number of channels in the output.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans

        self.layers = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_chans),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_chans),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Input 4D tensor of shape `(N, in_chans, H, W)`.

        Returns:
            Output tensor of shape `(N, out_chans, H, W)`.
        """
        return self.layers(image)


class GatedConvBlock(nn.Module):
    """
    Gated Convolutional Block.
    """

    def __init__(self, in_chans: int, out_chans: int):
        """
        Args:
            in_chans: Number of channels in the input.
            out_chans: Number of channels in the output.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans

        self.layers = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_chans),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        
        self.gatedlayers = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_chans),
            torch.sigmoid()
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Input 4D tensor of shape `(N, in_chans, H, W)`.

        Returns:
            Output tensor of shape `(N, out_chans, H, W)`.
        """
        x_img = self.layers(image)
        x_gate = self.gatedlayers(image)
        x = x_img * x_gate
        return x
    
    
class ConvBlockSM(nn.Module):
    """
    Stack of convs followed by spatial attention.
    """

    def __init__(self, in_chans=2, conv_num=0, out_chans = None, max_chans = None):
        """
        Args:
            in_chans: Number of channels in the input.
            out_chans: Number of channels in the output.
        """
        super().__init__()

        self.in_chans = in_chans
        self.chans_max = max_chans or in_chans 
        self.out_chans = out_chans or in_chans
        
        self.layers = []
        self.layers.append(nn.Sequential(nn.Conv2d(self.in_chans, self.chans_max, kernel_size=3, padding=1, bias=False),
                   nn.InstanceNorm2d(self.chans_max),
                   nn.LeakyReLU(negative_slope=0.2, inplace=True)))
        
        #### Spatial Attention ####
        self.SA = utils.SpatialAttention()

        for index in range(conv_num):
            if index == conv_num-1:
                self.layers.append(nn.Sequential(nn.Conv2d(self.chans_max, self.out_chans, kernel_size=3, padding=1, bias=False),
                       nn.InstanceNorm2d(self.in_chans),
                       nn.LeakyReLU(negative_slope=0.2, inplace=True)))
            else:
                self.layers.append(nn.Sequential(nn.Conv2d(self.chans_max, self.chans_max, kernel_size=3, padding=1, bias=False),
                       nn.InstanceNorm2d(self.chans_max),
                       nn.LeakyReLU(negative_slope=0.2, inplace=True)))
        
        self.body = nn.Sequential(*self.layers)


    def forward(self, image: torch.Tensor) -> torch.Tensor:

        output = self.body(image)
        output = output + image[:,:2,:,:]            
        #### Spatial Attention ###    
        output = self.SA(output)*output  
        
        return output
    
    
####  Transpose 2D ####
class TransposeConvBlock(nn.Module):
    """
    A Transpose Convolutional Block that consists of one convolution transpose
    layers followed by instance normalization and LeakyReLU activation.
    """

    def __init__(self, in_chans: int, out_chans: int):
        """
        Args:
            in_chans: Number of channels in the input.
            out_chans: Number of channels in the output.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans

        self.layers = nn.Sequential(
            nn.ConvTranspose2d(
                in_chans, out_chans, kernel_size=2, stride=2, bias=False
            ),
            nn.InstanceNorm2d(out_chans),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Input 4D tensor of shape `(N, in_chans, H, W)`.

        Returns:
            Output tensor of shape `(N, out_chans, H*2, W*2)`.
        """
        return self.layers(image)
        

class ConvBlockDown(nn.Module):
    """
    A Convolutional Block that decreases the dimension.
    """

    def __init__(self, in_chans, SR_scale):
        """
        Args:
            in_chans: Number of channels in the input.
        """
        super().__init__()

        self.in_chans = in_chans
        self.SR_scale =SR_scale
        self.layers = nn.Sequential(
            nn.Conv2d(in_chans, in_chans*SR_scale, kernel_size=SR_scale, stride=SR_scale, bias=False),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(in_chans*SR_scale, in_chans, kernel_size=3, padding=1, bias=False),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

    def forward(self, image):
        """
        Args:
            image: Input 4D tensor of shape `(N, in_chans, H, W)`.

        Returns:
            Output tensor of shape `(N, out_chans, H, W)`.
        """
        if self.SR_scale!=1:
            image = self.layers(image)
        return image
    
    
class ConvBlockUp(nn.Module):
    """
    A Convolutional Block that upscales the dimension.
    """

    def __init__(self, in_chans, SR_scale):
        """
        Args:
            in_chans: Number of channels in the input.
        """
        super().__init__()

        self.in_chans = in_chans
        self.SR_scale = SR_scale
        self.layers = nn.Sequential(
            nn.Conv2d(in_chans, in_chans, kernel_size=int(np.floor(SR_scale*1.5)), padding='same', bias=False),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(in_chans, in_chans, kernel_size=3, padding=1, bias=False),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        

    def forward(self, image):
        """
        Args:
            image: Input 4D tensor of shape `(N, in_chans, H, W)`.

        Returns:
            Output tensor of shape `(N, out_chans, H, W)`.
        """
        if self.SR_scale!=1:
            image = F.interpolate(image, scale_factor=self.SR_scale, mode='bicubic')
            image = self.layers(image)
        return image
    
