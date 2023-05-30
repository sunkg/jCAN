#!/usr/bin/env python-3
"""
@author: sunkg
"""
import os, sys, os.path, random
import time, statistics
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import torch
import numpy as np
import torch.utils.tensorboard
import torchvision
import torchvision.utils
import tqdm
from paired_dataset import get_paired_volume_datasets, center_crop
from basemodel import Config
from model import ReconModel
from augment import augment, flip
import utils
import torch.nn as nn

class Prefetch(torch.utils.data.Dataset):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = [i for i in tqdm.tqdm(dataset, leave=False)]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, ind):
        return self.dataset[ind]

def augment_None(batch):
    return batch

def augment_Rigid(batch):
    return [augment(x, rigid=True, bspline=False) for x in batch]

def augment_BSpline(batch):
    return [augment(x, rigid=True, bspline=True) for x in batch]

augment_funcs = { \
        'None': augment_None,
        'Rigid': augment_Rigid,
        'BSpline': augment_BSpline}

    
def main(args):
    # setup
    cfg = Config()
    cfg.sparsity_ref = args.sparsity_ref
    cfg.sparsity_tar = args.sparsity_tar
    cfg.lr = args.lr
    cfg.shape = args.shape
    cfg.img_size = tuple([cfg.shape, cfg.shape])
    cfg.coils = args.coils
    cfg.mask = args.mask
    cfg.use_amp = args.use_amp
    cfg.num_heads = args.num_heads 
    cfg.window_size = args.window_size 
    cfg.mlp_ratio = args.mlp_ratio 
    cfg.n_SC = args.n_SC
    cfg.num_recurrent = args.num_recurrent
    cfg.sens_chans = args.sens_chans 
    cfg.sens_steps = args.sens_steps
    cfg.embed_dim = args.embed_dim 
    cfg.lambda0 = args.lambda0
    cfg.lambda1 = args.lambda1
    cfg.lambda2 = args.lambda2
    cfg.lambda3 = args.lambda3
    cfg.GT = args.GT
    cfg.ds_ref = args.ds_ref
    cfg.protocals = args.protocals
    
    os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE' 
    
    if len(args.patch_size) == 1:
        cfg.patch_size = tuple([args.patch_size[0] for i in range(args.num_recurrent)])
    else:
        cfg.patch_size = tuple([args.patch_size[i%len(args.patch_size)] if i != args.num_recurrent-1 else 1 for i in range(args.num_recurrent)]) #Patch size is set as alternating 4,2,1.
    
    print(args)

    for path in [args.logdir, args.logdir+'/res', args.logdir+'/ckpt']:
        if not os.path.exists(path):
            os.mkdir(path)
            print('mkdir:', path)
    writer = torch.utils.tensorboard.SummaryWriter(args.logdir)

    print('loading model...')
    os.environ["CUBLAS_WORKSPACE_CONFIG"]= ':4096:8'
    #torch.use_deterministic_algorithms(True)
    seed = 14982321 
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cfg.device = device.type
    if cfg.device == 'cpu': cfg.GPUs = 0 
    else: cfg.GPUs = 1
    
    batchsize_train = args.batch_size
    iter_cnt = 0

    print('training from scratch...')

    net = ReconModel(cfg=cfg)
    epoch_start = 0

    net = net.to(device)

    writer.add_text('date', repr(time.ctime()))
    writer.add_text('working dir', repr(os.getcwd()))
    writer.add_text('__file__', repr(os.path.abspath(__file__)))
    writer.add_text('commands', repr(sys.argv))
    writer.add_text('arguments', repr(args))
    writer.add_text('actual config', repr(cfg))

    print('loading data...')
    volumes_train = get_paired_volume_datasets( \
            args.train, crop=int(args.shape*1.1), q=0, protocals=args.protocals, basepath = args.basepath, exchange_Modal = args.exchange_Modal) 
    volumes_val = get_paired_volume_datasets( \
            args.val, crop=cfg.shape, protocals=args.protocals, basepath = args.basepath, exchange_Modal = args.exchange_Modal)

    slices_val = torch.utils.data.ConcatDataset(volumes_val)

    #### For visualization during training ####
    len_vis = 16
    col_vis = 4
    batch_vis = next(iter(torch.utils.data.DataLoader( \
            slices_val, batch_size=len_vis, shuffle=True)))
    batch_vis = [x.to(device, non_blocking=True) for x in batch_vis]
    batch_vis = [utils.complex_to_chan_dim(x) for x in batch_vis]    

    print('training...')
    last_loss, last_ckpt, last_disp = 0, 0, 0
    time_data, time_vis = 0, 0
    signal_earlystop = False
    iter_best = iter_cnt
    Eval_PSNR_best = None
    Eval_SSIM_best = None

    optim_R = torch.optim.AdamW(net.parameters(), \
            lr=cfg.lr, weight_decay=0)
    
    scalar = torch.cuda.amp.GradScaler(enabled=cfg.use_amp)
    
    Eval_PSNR, Eval_SSIM = [], []
    time_start = time.time()
    
    for index_epoch in tqdm.trange(epoch_start, args.epoch, desc='epoch', leave=True):
    
        slices_train = torch.utils.data.ConcatDataset(volumes_train)
        
        if args.prefetch:
            # load all data to RAM
            slices_train = Prefetch(slices_train)
            slices_val = Prefetch(slices_val)

        print('dataset: ' \
                + str(len(slices_train)) + ' / ' \
                + str(len(volumes_train)) + ' for training, ' \
                + str(len(slices_val)) + ' / ' \
                + str(len(volumes_val)) + ' for validation')

        loader_train = torch.utils.data.DataLoader( \
            slices_train, batch_size=batchsize_train, shuffle=True, \
            num_workers=args.num_workers, pin_memory=True, drop_last=True)
        loader_val = torch.utils.data.DataLoader( \
            slices_val, batch_size=args.batch_size, shuffle=True, \
            num_workers=args.num_workers, pin_memory=True, drop_last=True)

        ###################  training ########################
        tqdm_iter = tqdm.tqdm(loader_train, desc='iter', \
                bar_format=str(batchsize_train)+': {n_fmt}/{total_fmt}'+\
                '[{elapsed}<{remaining},{rate_fmt}]'+'{postfix}', leave=False)
        
        #### learning rate decay ####
        if index_epoch%(args.epoch//3)==0:
            for param_group in optim_R.param_groups:
                param_group['lr'] = param_group['lr']*(0.5**(index_epoch//100))
                
        if signal_earlystop:
            break
        for batch in tqdm_iter:

            net.train()
            time_data = time.time() - time_start

            iter_cnt += 1
            with torch.no_grad():
                batch = [x.to(device, non_blocking=True) for x in batch]
                batch = augment_funcs[args.aux_aug](batch)
                batch = flip(batch, len(args.protocals))
                #batch = rotate(batch)
                batch = [center_crop(x, (cfg.shape, cfg.shape)) for x in batch]      
                batch = [utils.complex_to_chan_dim(x) for x in batch]            
            
            with torch.cuda.amp.autocast(enabled=cfg.use_amp):
                local_fidelities, loss_fidelity, loss_consistency, loss_ssim, loss_all = net(*batch)
            
            optim_R.zero_grad()
            scalar.scale(loss_all.mean()).backward()

            scalar.step(optim_R)
            scalar.update()
            
            del batch

            time_start = time.time()
  
            if iter_cnt % 5000 == 0:  # every 5000 iterations save recon images
                last_disp = iter_cnt
                net.eval()
                with torch.no_grad():
                    net.test(*batch_vis)
                    vis = net.get_vis('images')   
                for name, val in vis['images'].items():
                    torchvision.utils.save_image(val, \
                        args.logdir+'/res/'+'%010d_'%iter_cnt+name+'.jpg', \
                        nrow=len_vis//col_vis, padding=10, \
                        range=(0, 1), pad_value=0.5)
                del vis, name, val
            if (iter_cnt % 30000 == 0):  # every 30000 iterations save model param.
                last_ckpt = iter_cnt
                torch.save({'state_dict': net.state_dict(),
                            'config': cfg,
                            'epoch': index_epoch},
                           args.logdir+'/ckpt/ckpt_%010d.pt'%iter_cnt)

            time_vis = time.time() - time_start
            time_start = time.time()
            postfix = '[%d/%d/%d/%d]'%( \
                    iter_cnt, last_loss, last_disp, last_ckpt)
            if time_data >= 0.1:
                postfix += ' data %.1f'%time_data
            if time_vis >= 0.1:
                postfix += ' vis %.1f'%time_vis
            tqdm_iter.set_postfix_str(postfix)


        ###################  validation  ########################
        net.eval()

        tqdm_iter = tqdm.tqdm(loader_val, desc='iter', \
                bar_format=str(args.batch_size)+'(val) {n_fmt}/{total_fmt}'+\
                '[{elapsed}<{remaining},{rate_fmt}]'+'{postfix}', leave=False)
        stat_eval  = []
        stat_loss = []
        time_start = time.time()
        with torch.no_grad():
            for batch in tqdm_iter:
                time_data = time.time() - time_start
                batch = [x.to(device, non_blocking=True) for x in batch]
                batch = [utils.complex_to_chan_dim(x) for x in batch]   
               
                net.test(*batch)

                stat_loss.append(net.Eval)
                vis = net.get_vis('scalars')
                stat_eval.append(vis['scalars'])
                del batch

                time_start = time.time()
                if time_data >= 0.1:
                    postfix += ' data %.1f'%time_data
            vis = {key: statistics.mean([x[key] for x in stat_eval]) \
                    for key in stat_eval[0]}
            for name, val in vis.items():
                writer.add_scalar('val/'+name, val, iter_cnt)
            Eval_PSNR_current, Eval_SSIM_current = [(sum(i)/len(loader_val)) for i in zip(*stat_loss)]
            Eval_PSNR.append(Eval_PSNR_current)
            Eval_SSIM.append(Eval_SSIM_current)
            del vis 
 
            np.save(args.logdir+'/PSNR', np.array(Eval_PSNR))
            np.save(args.logdir+'/SSIM', np.array(Eval_SSIM))      

            if (Eval_PSNR_best is None) or ((Eval_PSNR_current > Eval_PSNR_best) & (Eval_SSIM_current > Eval_SSIM_best)):
                Eval_PSNR_best = Eval_PSNR_current
                Eval_SSIM_best = Eval_SSIM_current
                iter_best = iter_cnt
                print('Current best iteration %d/%d:'%(iter_best, len(loader_train)*args.epoch), f' PSNR: {Eval_PSNR_best:.2f}', f', SSIM: {Eval_SSIM_best:.4f}')
                torch.save({'state_dict': net.state_dict(),
                            'config': cfg,
                            'epoch': index_epoch},
                             args.logdir+'/ckpt/best.pt')  # save best model variant

            else:
                if iter_cnt >= args.early_stop + iter_best:
                    signal_earlystop=True
                    print('signal_earlystop set due to early_stop')
                
                      
    print('reached end of training loop, and signal_earlystop is '+str(signal_earlystop))
    writer.flush()
    writer.close()
    


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='CS with adaptive mask')
    parser.add_argument('--logdir', metavar='logdir', \
                        type=str, default='/public/bme/home/sunkc/fMRI/models',\
                        help='log directory')
    parser.add_argument('--epoch', type=int, default=300, \
                        help='epochs to train')
    parser.add_argument('--batch_size', type=int, default=1, \
                        help='batch size for training')
    parser.add_argument('--num_workers', type=int, default=8, \
                        help='number of threads for parallel preprocessing')
    parser.add_argument('--lr', type=float, default=5e-4, \
                        help='learning rate')
    parser.add_argument('--early_stop', type=int, default=1000000, metavar='N', \
                        help='stop training after val loss not going down for N iters')
    parser.add_argument('--n_SC', type=int, default=1, \
                        help='number of self-cross attention')
    parser.add_argument('--patch_size', type=tuple, default=tuple([4,2,1]), \
                        help='patch size in ViT')  
    parser.add_argument('--lambda0', type=float, default=10, \
                        help='weight of the kspace loss')
    parser.add_argument('--lambda1', type=float, default=10, \
                        help='weight of the consistency loss in K-space')
    parser.add_argument('--lambda2', type=float, default=1, \
                        help='weight of the SSIM loss')
    parser.add_argument('--lambda3', type=float, default=1e2, \
                        help='weight of the TV Loss')
    parser.add_argument('--embed_dim', type=int, default=32, \
                        help='dimension of embeddings in ViT')	
    parser.add_argument('--shape', type=int, default=320, \
                        help='image shape')					
    parser.add_argument('--num_heads', type=int, default=8, \
                        help='number of multiheads in ViT')
    parser.add_argument('--window_size', type=int, default=16, \
                        help='window size of the SwinTransformer')
    parser.add_argument('--num_recurrent', type=int, default=25, \
                        help='number of DCRBs')
    parser.add_argument('--mlp_ratio', type=int, default=32, \
                        help='ratio in MLP')
    parser.add_argument('--sens_chans', type=int, default=8, \
                        help='number of channels in sensitivity network')
    parser.add_argument('--sens_steps', type=int, default=4, \
                        help='number of steps in initial sensitivity network')
    parser.add_argument('--GT', type=bool, default=True, \
                        help='if there is GT, default is True') 
    parser.add_argument('--exchange_Modal', type=bool, default=False, \
                        help='exchange order of protocals for augmentation, default is False') 
    parser.add_argument('--ds_ref', type=bool, default=True, \
                        help='if use gradient map of reference image as input, default is True') 
    parser.add_argument('--mask', metavar='type', \
                        choices=['mask', 'taylor','lowpass', 'equispaced', 'loupe','random'], \
                        type=str, default = 'equispaced', help='types of mask')
    parser.add_argument('--sparsity_ref', metavar='0-1', \
                        type=float, default=1, help='sparisity of masks for reference modality')
    parser.add_argument('--sparsity_tar', metavar='0-1', \
                        type=float, default=0.25, help='sparisity of masks for target modality')
    parser.add_argument('--train', type=str, default='/public/bme/home/sunkc/fMRI/data/T1Flair_T2Flair_T2_train.csv', \
                        help='path to csv file of training data')
    parser.add_argument('--val', default='/public/bme/home/sunkc/fMRI/data/T1Flair_T2Flair_T2_val.csv', \
                        type=str, help='path to csv file of validation data')
    parser.add_argument('--basepath', default='/public_bme/share/DataBackup/UII_brain_T1F_T2F_T2/original_data', \
                            type=str, help='path to basepath of data')
    parser.add_argument('--coils', type=int, default=24, \
                        help='number of coils')
    parser.add_argument('--protocals', metavar='NAME', \
                        type=str, default= ['T2', 'T1Flair'], nargs='*', help='input modalities, first element is target, second is reference')
    parser.add_argument('--aux_aug', type=str, default = 'Rigid', \
                        choices=augment_funcs.keys(), help='data augmentation aux image')
    parser.add_argument('--prefetch', action='store_true')
    parser.add_argument('--use_amp', action='store_true')
    args = parser.parse_args()

    main(args)

