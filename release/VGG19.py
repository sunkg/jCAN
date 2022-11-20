# -*- coding: utf-8 -*-
"""
@author: sunkg
"""

import torch.nn as nn
import torch
import torchvision.models as models
import torch.nn.functional as F

####### VGG19 ########

class Vgg19(nn.Module):
    def __init__(self, indices, requires_grad=False):
        super(Vgg19, self).__init__()
        self.vgg_indices = indices
                
        # load pretrained VGG #
        self.vgg_model = models.vgg19(pretrained=False)    
        pthfile = r'/public/bme/home/sunkc/fMRI/pretrained/vgg19-dcbb9e9d.pth'
        self.vgg_model.load_state_dict(torch.load(pthfile))
        
        self.vgg_pretrained_features = self.vgg_model.features
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False
        
    def forward(self, X):
        out = []
        for i in range(self.vgg_indices[-1]):
            X = self.vgg_pretrained_features[i](X)
            if (i+1) in self.vgg_indices:
                out.append(X)
        return out


class VGGLoss(nn.Module):
    def __init__(self, indices=[16]):
        super(VGGLoss, self).__init__()
        self.vgg_loss_net = Vgg19(indices)
        self.vgg_loss_net.eval()
								
    def _vgg_preprocess(self, batch):
        mean = torch.zeros_like(batch)
        std = torch.zeros_like(batch)
        mean[:, 0, :, :] = 0.485
        mean[:, 1, :, :] = 0.456
        mean[:, 2, :, :] = 0.406
        std[:, 0, :, :] = 0.229
        std[:, 1, :, :] = 0.224
        std[:, 2, :, :] = 0.225
        
        batch = (batch - mean) / std
        
        return batch
 
 
    def forward(self, X, Y):
 
        X = self._vgg_preprocess(X)
        Y = self._vgg_preprocess(Y)
 
        feat_X = self.vgg_loss_net(X)
        feat_Y = self.vgg_loss_net(Y)
 
        vgg_loss = 0
        for j in range(len(feat_X)):
            vgg_loss += F.l1_loss(feat_X[j], feat_Y[j])
        return vgg_loss


if __name__ == '__main__':
	indices = [16]
	size = 320
	N, C = 3, 3
	img1 = torch.randn(N, C, size, size, dtype=torch.float)
	img2 = torch.randn(N, C, size, size, dtype=torch.float)
	VGGloss = VGGLoss(indices)
	loss =VGGloss(img1,img2)
	print(loss)
