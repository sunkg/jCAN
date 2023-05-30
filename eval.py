#!/usr/bin/env python3

import os, os.path, statistics, json
import torch
import numpy as np
import nibabel as nib
from paired_dataset import get_paired_volume_datasets, center_crop
from model import ReconModel
from augment import augment_eval
import torchvision.utils
from torchio.transforms import (
    RandomAffine,
    OneOf,
    Compose,
)

def augment_aux(batch, factor=1):
    assert factor > 0
    img_full, img_aux = batch
    _, grid =  augment_eval(img_aux, rigid=True, bspline=True)
    identity = np.array([[[1, 0, 0], [0, 1, 0]]])
    identity = identity * np.ones((img_aux.shape[0], 1, 1))
    identity = torch.as_tensor(identity, dtype=img_aux.abs().dtype).to(img_aux.device, non_blocking=True)
    identity = torch.nn.functional.affine_grid(identity, \
            size=img_aux.shape, align_corners=False)
    offset = grid - identity
    grid = identity + offset * factor
    img_aux, _ =  augment_eval(img_aux, rigid=False, bspline=False, grid=grid)
    return (img_full, img_aux)

def augment3D(coil_img, degree = ([0, 0.01, 0.01]), translation = ([0.05, 0, 0])): # through-plane rotation in degree, translation in mm
    img = torch.linalg.vector_norm(coil_img, ord=2, dim=1, keepdim=True) 
    sm = coil_img/img
    img = img.permute(1,0,2,3) #[1, slices, 320,320]
    transform = Compose([
    OneOf({
        RandomAffine(degrees = degree, translation = translation, scales = 0),
    })
    ])
    
    img_rigid = transform(img)
    img_rigid = img_rigid.permute(1,0,2,3) #[1, slices, 320,320]

    coil_img_rigid = sm*img_rigid

    return coil_img_rigid, img_rigid


def main(args):
    affine = np.eye(4)*[0.7,-0.7,-5,1]

    print(args)

    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)
            
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    if os.path.isfile(args.model_path) or os.path.isdir(args.model_path):
        ckpt = torch.load(args.model_path)
        cfg = ckpt['config']
        net = ReconModel(cfg=cfg)
        net.load_state_dict(ckpt['state_dict'])
        print('load ckpt from:', args.model_path)
    else:
        raise FileNotFoundError
        
    net.use_amp = False
    cfg = net.cfg
    net.GT = args.GT
 
    volumes = get_paired_volume_datasets( \
                args.val, crop=cfg.shape, protocals=args.protocals, basepath = args.basepath)
            
    net.eval()

    stat_eval = []
    PSNR, SSIM = [], []
    PSNR_raw, SSIM_raw= [], []
    col_vis = 4
    total = sum([param.nelement() for param in net.parameters()])
    print('Network size is %.2fM' % (total/1e6))
    
    for i, volume in enumerate(volumes):
        
        with torch.no_grad():
            if (args.aux_aug > 0) & (len(volume) > 1):
                volume_ref = torch.from_numpy(np.array(volume)[:,1,...])
                volume_ref_rigid, volume_ref_rss_rigid = augment3D(volume_ref, degree =([0, args.aux_aug*args.rotate, args.aux_aug*args.rotate]), translation = ([args.aux_aug*args.translate, 0, 0])) #through-plane motion
                #volume_ref_rigid, volume_ref_rss_rigid = augment3D(volume_ref, degree =([args.aux_aug*args.rotate 0, 0]), translation = ([0, args.aux_aug*args.translate, args.aux_aug*args.translate])) #in-plane motion
                volume_new = []
                for idx in range(len(volume)):
                    volume_new.append([volume[idx][0], volume_ref_rigid[idx].numpy()])
           
                batch = [torch.tensor(np.stack(s, axis=0)).to(device) for s in \
                    zip(*[volume_new[j] for j in range(len(volume_new))])]
            else:                
                batch = [torch.tensor(np.stack(s, axis=0)).to(device) for s in \
                    zip(*[volume[j] for j in range(len(volume))])]
            
            batch = [center_crop(i, (cfg.shape, cfg.shape)) for i in batch]

            net.test(*batch)

            vis = net.get_vis('scalars')
            stat_eval.append(vis['scalars'])
            PSNR.append(vis['scalars']['metric_PSNR'])
            SSIM.append(vis['scalars']['metric_SSIM'])
            PSNR_raw.append(vis['scalars']['metric_PSNR_raw'])
            SSIM_raw.append(vis['scalars']['metric_SSIM_raw'])
            print('Raw volume:',i+1, f', PSNR: {PSNR_raw[-1]:.2f}', f', SSIM: {SSIM_raw[-1]:.4f}' )
            print('Recon volume:',i+1, f', PSNR: {PSNR[-1]:.2f}', f', SSIM: {SSIM[-1]:.4f}' )

            vis = net.get_vis('images') 
            for name, val in vis['images'].items():
                torchvision.utils.save_image(val, \
                    args.save_path+'/'+'%010d_'%i+name+'.jpg', \
                    nrow=batch[0].shape[0]//col_vis, padding=10, \
                    range=(0, 1), pad_value=0.5)
            
            del batch            
    
        if args.save_img == False:
            continue
        image, sampled, aux, rec = net.Target_f_rss, net.Target_sampled_rss, net.Ref_f_rss, net.rec_rss
        image, sampled, aux, rec = [nib.Nifti1Image(x.cpu().squeeze(1).numpy().T, affine) for x in (image, sampled, aux, rec)]
        nib.save(image, args.save_path+'/'+str(i)+'_image.nii')
        nib.save(aux, args.save_path+'/'+str(i)+'_aux.nii')
        nib.save(sampled, args.save_path+'/'+str(i)+'_sampled.nii')
        nib.save(rec, args.save_path+'/'+str(i)+'_rec.nii')
    

    with open(args.save_path+'/'+os.path.split(args.model_path)[1][:-3]+'.txt', 'w') as f:
        json.dump(stat_eval, f)
    vis_mean = {key: statistics.mean([x[key] for x in stat_eval]) \
            for key in stat_eval[0]}
    vis_std = {key: statistics.stdev([x[key] for x in stat_eval]) \
            for key in stat_eval[0]}

    print(vis_mean)
    print(vis_std)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='jCAN for MRI reconstruction')
    parser.add_argument('--model_path', type=str, default=None, \
                        help='with ckpt path, set empty str to load latest ckpt')
    parser.add_argument('--save_path', type=str, default=None, \
                        help='path to save evaluated data')
    parser.add_argument('--save_img', default= True, \
                type=bool, help='save images or not')
    parser.add_argument('--val', default='/public/bme/home/sunkc/fMRI/data/T1Flair_T2Flair_T2_test.csv', \
            type=str, help='path to csv of test data')
    parser.add_argument('--basepath', default='/public_bme/share/DataBackup/UII_brain_T1F_T2F_T2/original_data', \
            type=str, help='path to test data')
    parser.add_argument('--shape', type=tuple, default=320, \
            help='mask and image shape, images will be cropped to match')
    parser.add_argument('--protocals', metavar='NAME', \
            type=str, default= ['T2','T1Flair'], nargs='*',
            help='input modalities')
    parser.add_argument('--aux_aug', type=float, default=-1, \
            help='data augmentation aux image, set to -1 to ignore')
    parser.add_argument('--rotate', type=float, default=0.01*180, \
            help='rotation augmentation in degree')  
    parser.add_argument('--translate', type=float, default=0.05, \
            help='translation augmentation in pixel')  
    parser.add_argument('--GT', type=bool, default=True, \
            help='if there is GT, default is True') 
    args = parser.parse_args()

    main(args)

