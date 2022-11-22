# -*- coding: utf-8 -*-
"""
@author: sunkg
"""
import torch
import torch.nn as nn
import ViT
import CNN
import utils
import sensitivity_model

class DCRB(nn.Module):
    def __init__(self, coils_all, img_size, num_heads, window_size, depth=2, patch_size =1,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, n_SC=2, embed_dim=96, ds_ref=True, SR_scale=1):
        super().__init__()
        
        self.stepsize = nn.Parameter(0.1*torch.rand(1))        
        self.LeakyReLU = nn.LeakyReLU()
        self.SR_scale = SR_scale
        self.img_size = img_size
        self.coils_all = coils_all
        self.ds_ref = ds_ref
        self.CNN = CNN.Unet(in_chans = coils_all*2, out_chans = coils_all, chans = 32) # using Unet for K-space, 2 is for real and imaginary channels         
        self.ViT = ViT.ViT(dim = 2, img_size=img_size, num_heads=num_heads, window_size=window_size, depth=depth, patch_size=patch_size,
                           mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop, attn_drop=attn_drop,
                           drop_path=drop_path, norm_layer=nn.LayerNorm, n_SC=n_SC, embed_dim=embed_dim, ds_ref=ds_ref) # each of T1 and T2 has two channels for real and imaginary values 
    
    
    def forward(self, Ref_img, Ref_Kspace_f, Target_Kspace_u, Target_img_f, mask, sens_maps_updated): 
        
        Target_Kspace_f = utils.sens_expand(Target_img_f, sens_maps_updated)     
        Ref_Kspace_f = utils.complex_to_chan_dim(Ref_Kspace_f)
        Target_Kspace_f = utils.complex_to_chan_dim(Target_Kspace_f)
        Ref_img = utils.complex_to_chan_dim(Ref_img)
        Target_img_f = utils.complex_to_chan_dim(Target_img_f)
            
        #### integrate gradient map into input ####
        if self.ds_ref:
            Ref_rss = utils.rss(Ref_img)
            Ref_grad = utils.gradient(Ref_rss)
            Ref_img = torch.cat([Ref_img, Ref_grad], dim=1)
   
        input_CNN = torch.cat([Ref_Kspace_f, Target_Kspace_f], 1)
        input_ViT = [Target_img_f, Ref_img]
        output_CNN = self.CNN(input_CNN)
        output_ViT = self.ViT(input_ViT[0], input_ViT[1])
        
        #### denormalize and turn back to complex values #### 
        output_CNN = utils.chan_dim_to_complex(output_CNN)      
        output_ViT = utils.chan_dim_to_complex(output_ViT)
        Target_img_f = utils.chan_dim_to_complex(Target_img_f)        
        
        Target_Kspace_f_down = utils.sens_expand(Target_img_f, sens_maps_updated)        
        term1 = 2*utils.sens_reduce(mask*(mask*Target_Kspace_f_down-Target_Kspace_u), sens_maps_updated) 
        term2 = utils.sens_reduce(output_CNN, sens_maps_updated)
        
        #### scaling wit constant 0.1 ####
        Target_img_f = Target_img_f-self.stepsize*(term1+0.1*term2+0.1*output_ViT)
        
        return Target_img_f
        

class fD2RT(nn.Module):
    def __init__(self, coils, img_size, num_heads, window_size, depth=2, patch_size = None,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, n_SC=2, num_recurrent=5, embed_dim=96, 
                 sens_chans=8, sens_steps=4, ds_ref=True, SR_scale=1, is_Unet=True):
        super().__init__()
        
        self.sens_net = sensitivity_model.SensitivityModel(
            chans = sens_chans,
            sens_steps = sens_steps,
            is_Unet = is_Unet,
        )
        self.ds_ref = ds_ref
        self.SR_scale = SR_scale 
        self.coils = coils//2 # coils of single modality
        self.stepsize = nn.Parameter(0.1*torch.rand(1))
        self.num_recurrent = num_recurrent
        self.recurrent = nn.ModuleList([DCRB(coils_all=coils, img_size=img_size, num_heads=num_heads, window_size=window_size, depth=depth, 
                         patch_size = int(patch_size[i]), mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop, 
                         attn_drop=attn_drop, drop_path=drop_path, norm_layer=nn.LayerNorm, n_SC=n_SC, embed_dim=embed_dim, 
                         ds_ref= ds_ref, SR_scale=SR_scale) for i in range(num_recurrent)])
        
        #### CNN in SMRB ####
        self.ConvBlockSM = nn.ModuleList([CNN.ConvBlockSM(in_chans = 2, conv_num = 2) for _ in range(num_recurrent-1)])
    
    #### SMRB ####
    def SMR(self, Target_img_f, sens_maps_updated, Target_Kspace_u, mask, gate, idx):
        Target_Kspace_f = utils.sens_expand(Target_img_f, sens_maps_updated)    
        B, C, H, W = sens_maps_updated.shape
        sens_maps_updated_ = sens_maps_updated.reshape(B*C, 1, H, W)
        sens_maps_updated_ = utils.complex_to_chan_dim(sens_maps_updated_)    
        sens_maps_updated_ = self.ConvBlockSM[idx](sens_maps_updated_)
        sens_maps_updated_ = utils.chan_dim_to_complex(sens_maps_updated_) 
        sens_maps_updated_ = sens_maps_updated_.reshape(B, C, H, W)
        sens_maps_updated = sens_maps_updated - self.stepsize*(2*utils.ifft2(mask*(mask*Target_Kspace_f - Target_Kspace_u) * Target_img_f.transpose(-2, -1).conj()) + 0.1*sens_maps_updated_)         
        sens_maps_updated = sens_maps_updated / (utils.rss(sens_maps_updated) + 1e-12)
        sens_maps_updated = sens_maps_updated * gate
        return sens_maps_updated    

            
    def forward(self, Ref_Kspace_f, Target_Kspace_u, mask, num_low_frequencies):
        rec = []
        
        #### SMEB ####
        if self.coils == 1:
            sens_maps = torch.ones_like(Target_Kspace_u)
            sens_maps_updated = torch.ones_like(Target_Kspace_u)
        else:
            sens_maps, gate = self.sens_net(Target_Kspace_u, num_low_frequencies)
            sens_maps_updated = sens_maps.clone()
      
        
        if Ref_Kspace_f != None:
            Ref_img = utils.sens_reduce(Ref_Kspace_f, sens_maps)
             
        Target_img_f = utils.sens_reduce(Target_Kspace_u, sens_maps) # initialization of Target image
        
        #### DCRB blocks #### 
        for idx, DCRB_ in enumerate(self.recurrent):
            if (self.coils != 1) & (idx != 0):            
                sens_maps_updated = self.SMR(Target_img_f, sens_maps_updated, Target_Kspace_u, mask, gate, idx-1)            

            if Ref_Kspace_f == None:
                Ref_img = Target_img_f.clone()
                Ref_Kspace_f = utils.sens_expand(Ref_img, sens_maps_updated)
                
            Target_img_f = DCRB_(Ref_img, Ref_Kspace_f, Target_Kspace_u, Target_img_f, mask, sens_maps_updated)
            rec.append(Target_img_f)
         
        return rec, utils.rss(Target_img_f), sens_maps_updated, Target_img_f
