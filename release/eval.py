#!/usr/bin/env python3

import os, os.path, statistics, json
import torch
import numpy as np
import nibabel as nib
from paired_dataset import get_paired_volume_datasets, center_crop
from model import ReconModel
from augment import augment
import torchvision.utils
import utils
from ptflops import get_model_complexity_info

def augment_aux(batch, factor=1):
    assert factor > 0
    img_full, img_aux = batch
    _, grid =  augment(img_aux, rigid=True, bspline=True)
    identity = np.array([[[1, 0, 0], [0, 1, 0]]])
    identity = identity * np.ones((img_aux.shape[0], 1, 1))
    identity = torch.as_tensor(identity, dtype=img_aux.abs().dtype).to(img_aux.device, non_blocking=True)
    identity = torch.nn.functional.affine_grid(identity, \
            size=img_aux.shape, align_corners=False)
    offset = grid - identity
    grid = identity + offset * factor
    img_aux, _ =  augment(img_aux, rigid=False, bspline=False, grid=grid)
    return (img_full, img_aux)

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
 
    if args.aux_aug > 0:
        volumes = get_paired_volume_datasets( \
                args.val, crop=int(cfg.shape*1.05), protocals=args.protocals, basepath = args.basepath)
    else:
        volumes = get_paired_volume_datasets( \
                args.val, crop=cfg.shape, protocals=args.protocals, basepath = args.basepath)
            
    net.eval()

    stat_eval = []
    PSNR = []
    SSIM = []
    PSNR_raw = []
    SSIM_raw = []
    col_vis = 4
    for i, volume in enumerate(volumes):
        with torch.no_grad():
            batch = [torch.tensor(np.stack(s, axis=0)).to(device) for s in \
                    zip(*[volume[j] for j in range(len(volume))])]
            if args.aux_aug > 0:
                img_full, img_aux = batch
                batch =  augment_aux(batch, args.aux_aug)
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
        image, sampled, aux, rec = net.img_Target_f_rss, net.img_Target_sampled_rss, net.img_Ref_f_rss, net.img_rec_rss
        image, sampled, aux, rec = [nib.Nifti1Image(x.cpu().squeeze(1).numpy().T, affine) for x in (image, sampled, aux, rec)]
        nib.save(image, args.save_path+'/'+str(i)+'_image.nii')
        nib.save(aux, args.save_path+'/'+str(i)+'_aux.nii')
        nib.save(sampled, args.save_path+'/'+str(i)+'_sampled.nii')
        nib.save(rec, args.save_path+'/'+str(i)+'_rec.nii')

    B, C, H, W = net.sens_maps.shape
    for i in range(len(net.SMs)):
        np.save(args.save_path+'/sensitivitymap'+str(i), \
                utils.rss(net.SMs[i].view(-1, 1, H , W)).cpu().detach().numpy())
        np.save(args.save_path+'/recs'+str(i), \
                utils.rss(net.recs_complex[i].view(-1, 1, H , W)).cpu().detach().numpy())   


    with open(args.save_path+'/'+os.path.split(args.model_path)[1][:-3]+'.txt', 'w') as f:
        json.dump(stat_eval, f)
    vis_mean = {key: statistics.mean([x[key] for x in stat_eval]) \
            for key in stat_eval[0]}
    vis_std = {key: statistics.stdev([x[key] for x in stat_eval]) \
            for key in stat_eval[0]}
    np.save(args.save_path+'/'+os.path.split(args.model_path)[1][:-3]+'_PSNR', np.array(PSNR))
    np.save(args.save_path+'/'+os.path.split(args.model_path)[1][:-3]+'_SSIM', np.array(SSIM))
    np.save(args.save_path+'/'+os.path.split(args.model_path)[1][:-3]+'_PSNR_raw', np.array(PSNR_raw))
    np.save(args.save_path+'/'+os.path.split(args.model_path)[1][:-3]+'_SSIM_raw', np.array(SSIM_raw))
    print(vis_mean)
    print(vis_std)

    with torch.cuda.device(0):
        macs, params = get_model_complexity_info(net, (48, 320, 320), as_strings=True,
                                           print_per_layer_stat=True, verbose=True)
        print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        print('{:<30}  {:<8}'.format('Number of parameters: ', params))

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='jCAN')
    parser.add_argument('--model_path', type=str, default=None, \
                        help='with ckpt path')
    parser.add_argument('--save_path', type=str, default=None, \
                        help='path to save evaluated data')
    parser.add_argument('--save_img', default= True, \
                type=bool, help='save images or not')
    parser.add_argument('--val', default='/xxx.csv', \
            type=str, help='path to csv of test data')
    parser.add_argument('--basepath', default='/xxx', \
            type=str, help='path to test files')
    parser.add_argument('--shape', type=tuple, default=320, \
            help='Image shape, images will be cropped to match')
    parser.add_argument('--protocals', metavar='NAME', \
            type=str, default= ['T2','T1Flair'], nargs='*',
            help='input modalities')
    parser.add_argument('--aux_aug', type=float, default=-1, \
            help='Data augmentation aux image, set to -1 to ignore')
    parser.add_argument('--GT', type=bool, default=True, \
            help='GT exists, default is True') 
    args = parser.parse_args()

    main(args)

