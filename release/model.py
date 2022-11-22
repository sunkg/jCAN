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
import VGG19
from utils import rss, fft2, ifft2
import utils


def generate_rhos(num_recurrent):
    rhos = [0.9**i for i in range(num_recurrent-1,-1,-1)]
    return rhos


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
        self.num_low_frequencies = int(self.cfg.img_size[1]*self.cfg.sparsity*0.32//self.cfg.SR_scale)

        if self.cfg.mask in ["mask", "taylor"]:
            self.net_mask = masks[self.cfg.mask](self.cfg.img_size[1]//self.cfg.SR_scale).to(self.device) # the number of columns 
        else:
            self.net_mask = masks[self.cfg.mask](self.cfg.sparsity, self.cfg.img_size[1]//self.cfg.SR_scale).to(self.device) # the number of columns 
                                    
        self.net_R = fD2RT.fD2RT(coils=self.cfg.coils*2, img_size=self.cfg.img_size, 
                           num_heads=self.cfg.num_heads, window_size=self.cfg.window_size, patch_size = self.cfg.patch_size, 
                           depth=2, mlp_ratio=self.cfg.mlp_ratio, qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., 
                           drop_path=0., norm_layer=nn.LayerNorm, n_SC=self.cfg.n_SC, num_recurrent=self.cfg.num_recurrent, 
                           embed_dim=self.cfg.embed_dim, sens_chans=self.cfg.sens_chans, sens_steps=self.cfg.sens_steps, 
                           ds_ref=self.cfg.ds_ref, SR_scale=self.cfg.SR_scale, is_Unet=self.cfg.is_Unet).to(self.device) # 2*coils for T1 and T2
        
        self.VGGLoss = VGG19.VGGLoss(self.cfg.vgg_indices).to(self.device)

        
    def set_input_noGT(self, Target_Kspace_sampled, Ref_Kspace_f=None):

        B, C, H, W = self.Target_Kspace_sampled.shape
        self.img_Target_f_rss = torch.zeros([B, 1, H*self.cfg.SR_scale, W*self.cfg.SR_scale], dtype=torch.complex64)
        self.Target_Kspace_f = torch.zeros([B, C, H*self.cfg.SR_scale, W*self.cfg.SR_scale], dtype=torch.complex64)
        
        if not self.cfg.is_Kspace:
            self.img_Target_Kspace_sampled = fft2(Target_Kspace_sampled)
            if Ref_Kspace_f is None:
                self.Ref_Kspace_f = None
                self.img_Ref_f_rss = torch.ones_like(self.img_Target_f_rss)
            else:
                self.img_Ref_f_rss = rss(Ref_Kspace_f)
                self.Ref_Kspace_f = fft2(Ref_Kspace_f)   
                
        else:    
            self.img_Target_Kspace_sampled = Target_Kspace_sampled
            if Ref_Kspace_f is None:
                self.Ref_Kspace_f = None
                self.img_Ref_f_rss = torch.ones_like(self.img_Target_f_rss)
            else:
                self.Ref_Kspace_f = Ref_Kspace_f 
                self.img_Ref_f_rss = rss(ifft2(self.Ref_Kspace_f))
                        
        self.img_Target_sampled_rss = rss(ifft2(self.img_Target_Kspace_sampled))
        
        with torch.no_grad(): # avoid update of mask
            self.mask = torch.logical_not(self.net_mask.pruned)


    def set_input_GT(self, Target_Kspace_f, Ref_Kspace_f=None):
        
        if not self.cfg.is_Kspace:
            self.img_Target_f_rss = rss(Target_Kspace_f)
            self.Target_Kspace_f = fft2(Target_Kspace_f)
            if Ref_Kspace_f is None:
                self.Ref_Kspace_f = None
                self.img_Ref_f_rss = torch.ones_like(self.img_Target_f_rss)
            else:
                self.img_Ref_f_rss = rss(Ref_Kspace_f)
                self.Ref_Kspace_f = fft2(Ref_Kspace_f) 

        else:    
            self.Target_Kspace_f = Target_Kspace_f
            self.img_Target_f_rss = rss(ifft2(self.Target_Kspace_f))
            if Ref_Kspace_f is None:
                self.Ref_Kspace_f = None
                self.img_Ref_f_rss = torch.ones_like(self.img_Target_f_rss)
            else:
                self.Ref_Kspace_f = Ref_Kspace_f 
                self.img_Ref_f_rss = rss(ifft2(self.Ref_Kspace_f))
            
        with torch.no_grad(): # avoid update of mask
            self.mask = torch.logical_not(self.net_mask.pruned)
            self.img_Target_Kspace_sampled = self.Target_Kspace_f * self.mask
     
        self.img_Target_sampled_rss = rss(ifft2(self.img_Target_Kspace_sampled))
            

    def forward(self, Target_Kspace_f, Ref_Kspace_f=None):
        if not torch.is_complex(Target_Kspace_f):
            Target_Kspace_f = utils.chan_dim_to_complex(Target_Kspace_f)
            if Ref_Kspace_f!=None:
                Ref_Kspace_f = utils.chan_dim_to_complex(Ref_Kspace_f)
                
        with torch.cuda.amp.autocast(enabled=self.cfg.use_amp):        
            if self.cfg.GT==True:
                self.set_input_GT(Target_Kspace_f, Ref_Kspace_f)
            else:
                self.set_input_noGT(Target_Kspace_f, Ref_Kspace_f)
                
            self.recs_complex, self.img_rec_rss, self.sens_maps, self.rec_down = self.net_R(\
                    Ref_Kspace_f = self.Ref_Kspace_f,
                    Target_Kspace_u = self.img_Target_Kspace_sampled,
                    mask = self.mask,
                    num_low_frequencies = self.num_low_frequencies
                    )
            # For loss record
            self.loss_all = 0
            self.loss_fidelity = 0
            self.local_fidelities = []
            
            for i in range(self.cfg.num_recurrent):
                loss_fidelity = F.l1_loss(rss(self.recs_complex[i]),self.img_Target_f_rss)+self.cfg.alpha*F.l1_loss(utils.sens_expand(self.recs_complex[i], utils.UpImgComplex(self.sens_maps, self.cfg.SR_scale)), self.Target_Kspace_f)
                self.local_fidelities.append(self.rhos[i]*loss_fidelity)
                self.loss_fidelity += self.local_fidelities[-1]
    
            self.loss_all += self.loss_fidelity
    
            #### DC loss ###
            self.loss_consistency = self.cfg.beta*F.l1_loss(self.mask*utils.sens_expand(self.rec_down,self.sens_maps), self.img_Target_Kspace_sampled)
            self.loss_all += self.loss_consistency
            
            #### VGG loss ###
            vgg_img1 = torch.cat([self.img_rec_rss]*3,1)
            vgg_img2 = torch.cat([self.img_Target_f_rss]*3,1)
            self.loss_VGG = self.cfg.vgg_lambdas*self.VGGLoss(vgg_img1,vgg_img2)
            self.loss_all += self.loss_VGG
            
            #### gradient loss ###
            self.rec_grad = utils.gradient(self.img_rec_rss)
            self.target_grad = utils.gradient(self.img_Target_f_rss)
            self.loss_grad = self.cfg.gamma*F.l1_loss(self.rec_grad, self.target_grad)   ##default 0.5
            self.loss_all += self.loss_grad
                 
            return self.local_fidelities, self.loss_fidelity, self.loss_consistency, self.loss_VGG, self.loss_all
        

    def test(self, Target_Kspace_f, Ref_Kspace_f=None):
        assert self.training == False
        with torch.cuda.amp.autocast(enabled=self.cfg.use_amp):
            with torch.no_grad():
                self.forward(Target_Kspace_f, Ref_Kspace_f)
                self.metric_PSNR = metrics.psnr(self.img_Target_f_rss, self.img_rec_rss, self.cfg.GT)
                self.metric_SSIM = metrics.ssim(self.img_Target_f_rss, self.img_rec_rss, self.cfg.GT)
                self.metric_MAE = metrics.mae(self.img_Target_f_rss, self.img_rec_rss, self.cfg.GT)
                self.metric_MSE = metrics.mse(self.img_Target_f_rss, self.img_rec_rss, self.cfg.GT)
                
                self.metric_PSNR_raw = metrics.psnr(self.img_Target_f_rss, \
                                       nn.functional.interpolate(self.img_Target_sampled_rss, scale_factor=self.cfg.SR_scale, mode='bicubic'), self.cfg.GT)
                self.metric_SSIM_raw = metrics.ssim(self.img_Target_f_rss, \
                                       nn.functional.interpolate(self.img_Target_sampled_rss, scale_factor=self.cfg.SR_scale, mode='bicubic'), self.cfg.GT)
                self.Eval = tuple([self.metric_PSNR, self.metric_SSIM])


    def prune(self, *args, **kwargs):
        assert False, 'Take care of amp'
        return self.net_mask.prune(*args, **kwargs)

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
                    lambda x: x.startswith('img_'), self.__dict__.keys()):
                image_val = getattr(self, image_name)
                if (image_val is not None) \
                        and (image_val.shape[1]==1 or image_val.shape[1]==3) \
                        and not torch.is_complex(image_val):
                    vis['images'][image_name] = image_val.detach()
        if content == 'histograms' or content is None:
            vis['histograms'] = {}
            if self.net_mask.weight is not None:
                vis['histograms']['weights'] = { \
                        'values': self.net_mask.weight.detach()}
        return vis
