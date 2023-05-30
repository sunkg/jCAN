# -*- coding: utf-8 -*-
"""
@author: sunkg
"""
import torch
import torch.fft
import torch.nn.functional as F
from masks import Mask, LowpassMask, EquispacedMask, LOUPEMask, TaylorMask, RandomMask
from basemodel import BaseModel
import metrics
import torch.nn as nn
import fD2RT
from utils import rss, fft2, ifft2, ssimloss
import utils

def gradient_loss(s):
    assert s.shape[-1] == 2, 'not 2D grid?'
    dx = torch.abs(s[:, :, 1:, :] - s[:, :, :-1, :])
    dy = torch.abs(s[:, 1:, :, :] - s[:, :-1, :, :])
    dy = dy*dy
    dx = dx*dx
    d = torch.mean(dx)+torch.mean(dy)
    return d/2.0

def generate_rhos(num_recurrent):
    rhos = [0.85**i for i in range(num_recurrent-1,-1,-1)]
    return rhos


def TV_loss(img, weight):
     bs_img, c_img, h_img, w_img = img.size()
     tv_h = torch.pow(img[:,:,1:,:]-img[:,:,:-1,:], 2).sum()
     tv_w = torch.pow(img[:,:,:,1:]-img[:,:,:,:-1], 2).sum()
     return weight*(tv_h+tv_w)/(bs_img*c_img*h_img*w_img)


masks = {"mask": Mask,
        "taylor": TaylorMask,
        "random": RandomMask,
        "lowpass": LowpassMask,
        "equispaced": EquispacedMask,
        "loupe": LOUPEMask}

    
class ReconModel(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.rhos = generate_rhos(self.cfg.num_recurrent)
        self.device = self.cfg.device
        self.num_low_frequencies = int(self.cfg.img_size[1]*self.cfg.sparsity_tar*0.32)  # 32% acquired lines in the ACS

        self.net_mask_ref = masks[self.cfg.mask](self.cfg.sparsity_ref, self.cfg.img_size[1]).to(self.device) # the number of columns 
        self.net_mask_tar = masks[self.cfg.mask](self.cfg.sparsity_tar, self.cfg.img_size[1]).to(self.device) # the number of columns 

                                    
        self.net_R = fD2RT.fD2RT(coils=self.cfg.coils*2, img_size=self.cfg.img_size, 
                           num_heads=self.cfg.num_heads, window_size=self.cfg.window_size, patch_size = self.cfg.patch_size, 
                           mlp_ratio=self.cfg.mlp_ratio, qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., 
                           drop_path=0., norm_layer=nn.LayerNorm, n_SC=self.cfg.n_SC, num_recurrent=self.cfg.num_recurrent, 
                           embed_dim=self.cfg.embed_dim, sens_chans=self.cfg.sens_chans, sens_steps=self.cfg.sens_steps, 
                           ds_ref=self.cfg.ds_ref).to(self.device) # 2*coils for target and reference
        
        
    def set_input_noGT(self, Target_img_sampled, Ref_img_f=None):

        B, C, H, W = self.Target_img_sampled.shape
        self.Target_f_rss = torch.zeros([B, 1, H, W], dtype=torch.complex64)
        self.Target_Kspace_f = torch.zeros([B, C, H, W], dtype=torch.complex64)
        
        self.Target_Kspace_sampled = fft2(Target_img_sampled)
        if Ref_img_f is None:
            self.Ref_Kspace_f = None
            self.Ref_f_rss = torch.ones_like(self.Target_f_rss)
        else:
            self.Ref_f_rss = rss(Ref_img_f)
            self.Ref_Kspace_f = fft2(Ref_img_f)                  
                        
        self.Target_sampled_rss = rss(ifft2(self.Target_Kspace_sampled))
        
        with torch.no_grad(): # avoid update of mask
            self.mask_ref = torch.logical_not(self.net_mask_ref.pruned)
            self.mask_tar = torch.logical_not(self.net_mask_tar.pruned)


    def set_input_GT(self, Target_img_f, Ref_img_f = None):
        
        self.Target_f_rss = rss(Target_img_f)
        self.Target_Kspace_f = fft2(Target_img_f)
        if Ref_img_f is None:
            self.Ref_img_f = None
            self.Ref_f_rss = torch.ones_like(self.Target_f_rss)
        else:
            self.Ref_f_rss = rss(Ref_img_f)
            self.Ref_Kspace_f = fft2(Ref_img_f) 
            
        with torch.no_grad(): # avoid update of mask
            self.mask_ref = torch.logical_not(self.net_mask_ref.pruned)
            self.mask_tar = torch.logical_not(self.net_mask_tar.pruned)
            self.Target_Kspace_sampled = self.Target_Kspace_f * self.mask_tar
            if self.Ref_Kspace_f != None:
                self.Ref_Kspace_sampled = self.Ref_Kspace_f * self.mask_ref 
     
        self.Target_sampled_rss = rss(ifft2(self.Target_Kspace_sampled))
            

    def forward(self, Target_img_f, Ref_img_f = None):
        if not torch.is_complex(Target_img_f):
            Target_img_f = utils.chan_dim_to_complex(Target_img_f)
            if Ref_img_f != None:
                Ref_img_f = utils.chan_dim_to_complex(Ref_img_f)
                
        with torch.cuda.amp.autocast(enabled=self.cfg.use_amp):        
            if self.cfg.GT == True:
                self.set_input_GT(Target_img_f, Ref_img_f)
            else:
                self.set_input_noGT(Target_img_f, Ref_img_f)
                
            self.recs_complex, self.rec_rss, self.sens_maps, self.rec_img = self.net_R(\
                    Ref_Kspace_f = self.Ref_Kspace_sampled,
                    Target_Kspace_u = self.Target_Kspace_sampled,
                    mask = self.mask_tar,
                    num_low_frequencies = self.num_low_frequencies
                    )

            # For loss record
            self.loss_all = 0
            self.loss_fidelity = 0
            self.local_fidelities = []
            for i in range(self.cfg.num_recurrent):
                loss_fidelity = F.l1_loss(rss(self.recs_complex[i]),self.Target_f_rss)+self.cfg.lambda0*F.l1_loss(utils.sens_expand(self.recs_complex[i], self.sens_maps), self.Target_Kspace_f)
                self.local_fidelities.append(self.rhos[i]*loss_fidelity)
                self.loss_fidelity += self.local_fidelities[-1]
    
            self.loss_all += self.loss_fidelity
    
            self.loss_consistency = self.cfg.lambda1*F.l1_loss(self.mask_tar*utils.sens_expand(self.rec_img, self.sens_maps), self.Target_Kspace_sampled)

            self.loss_all += self.loss_consistency
            
            self.loss_TV = TV_loss(torch.abs(self.sens_maps), self.cfg.lambda3)
            self.loss_all += self.loss_TV
     
            self.loss_ssim = self.cfg.lambda2 * ssimloss(self.rec_rss, self.Target_f_rss)
            self.loss_all += self.loss_ssim
     
            return self.local_fidelities, self.loss_fidelity, self.loss_consistency, self.loss_ssim, self.loss_all
        

    def test(self, Target_img_f, Ref_img_f=None):
        assert self.training == False
        with torch.cuda.amp.autocast(enabled=self.cfg.use_amp):
            with torch.no_grad():
                self.forward(Target_img_f, Ref_img_f)
                self.metric_PSNR = metrics.psnr(self.Target_f_rss, self.rec_rss, self.cfg.GT)
                self.metric_SSIM = metrics.ssim(self.Target_f_rss, self.rec_rss, self.cfg.GT)
                self.metric_MAE = metrics.mae(self.Target_f_rss, self.rec_rss, self.cfg.GT)
                self.metric_MSE = metrics.mse(self.Target_f_rss, self.rec_rss, self.cfg.GT)
                
                self.metric_PSNR_raw = metrics.psnr(self.Target_f_rss, self.Target_sampled_rss, self.cfg.GT)
                self.metric_SSIM_raw = metrics.ssim(self.Target_f_rss, self.Target_sampled_rss, self.cfg.GT)
                self.Eval = tuple([self.metric_PSNR, self.metric_SSIM])


    def prune(self, *args, **kwargs):
        assert False, 'Take care of amp'
        return self.net_mask_tar.prune(*args, **kwargs)

    def get_vis(self, content=None):
        assert content in [None, 'scalars', 'histograms', 'images']
        vis = {}
        if content == 'scalars' or content is None:
            vis['scalars'] = {}
            for loss_name in filter( \
                    lambda x: x.startswith('loss_'), self.__dict__.keys()):
                loss_val = getattr(self, loss_name)
                if loss_val is not None:
                    vis['scalars'][loss_name] = loss_val.detach().item()
            for metric_name in filter( \
                    lambda x: x.startswith('metric_'), self.__dict__.keys()):
                metric_val = getattr(self, metric_name)
                if metric_val is not None:
                    vis['scalars'][metric_name] = metric_val
        if content == 'images' or content is None:
            vis['images'] = {}
            for image_name in filter( \
                    lambda x: x.endswith('_rss'), self.__dict__.keys()):
                image_val = getattr(self, image_name)
                if (image_val is not None) \
                        and (image_val.shape[1]==1 or image_val.shape[1]==3) \
                        and not torch.is_complex(image_val):
                    vis['images'][image_name] = image_val.detach()
        if content == 'histograms' or content is None:
            vis['histograms'] = {}
            if self.net_mask_tar.weight is not None:
                vis['histograms']['weights'] = { \
                        'values': self.net_mask_tar.weight.detach()}
        return vis
