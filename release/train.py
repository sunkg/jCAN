#!/usr/bin/env python-3
"""
@author: sunkg
"""
import os, sys, os.path
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
from augment import augment, flip, rotate
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
    cfg.sparsity = args.sparsity
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
    cfg.vgg_indices = args.vgg_indices
    cfg.vgg_lambdas = args.vgg_lambdas
    cfg.sens_chans = args.sens_chans 
    cfg.sens_steps = args.sens_steps
    cfg.embed_dim = args.embed_dim 
    cfg.patch_size = args.patch_size
    cfg.beta = args.beta
    cfg.gamma = args.gamma
    cfg.alpha = args.alpha
    cfg.GT = args.GT
    cfg.ds_ref = args.ds_ref
    cfg.SR_scale = args.SR_scale
    cfg.is_Kspace = args.is_Kspace
    cfg.is_Unet = args.is_Unet
    cfg.protocals = args.protocals
    
    os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE' 
    
    if len(cfg.patch_size)!=cfg.num_recurrent:
        cfg.patch_size = tuple([1 for _ in range(args.num_recurrent)])
        print('Patch size is set as 1 since not given explicitly.')
    
    print(args)
    for path in [args.logdir, args.logdir+'/res', args.logdir+'/ckpt']:
        if not os.path.exists(path):
            os.mkdir(path)
            print('mkdir:', path)
    writer = torch.utils.tensorboard.SummaryWriter(args.logdir)

    print('loading model...')
    #seed = 19950102+666+233
    #torch.manual_seed(seed)
    #np.random.seed(seed)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cfg.device = device.type
    if cfg.device == 'cpu': cfg.GPUs = 0 
    else: cfg.GPUs = 1
    
    batchsize_train = args.batch_size
    iter_cnt = 0

    print('training from scratch...')
    net = ReconModel(cfg=cfg)
    epoch_start = 0

  ######## Use parallel training if possible #####
    if torch.cuda.device_count() > 1:
        cnt_GPU = torch.cuda.device_count()
        cfg.GPUs = cnt_GPU
        print("Let's use", cnt_GPU, "GPUs!")
        print(net.cfg)
        batchsize_train = cnt_GPU*args.batch_size
        net = nn.DataParallel(net)

    net = net.to(device)

    writer.add_text('date', repr(time.ctime()))
    writer.add_text('working dir', repr(os.getcwd()))
    writer.add_text('__file__', repr(os.path.abspath(__file__)))
    writer.add_text('commands', repr(sys.argv))
    writer.add_text('arguments', repr(args))
    writer.add_text('actual config', repr(cfg))

    print('loading data...')
    volumes_train = get_paired_volume_datasets( \
            args.train, crop=int(args.shape*1.1), protocals=args.protocals, basepath = args.basepath, exchange_Modal = args.exchange_Modal)
    volumes_val = get_paired_volume_datasets( \
            args.val, crop=cfg.shape, protocals=args.protocals, basepath = args.basepath, exchange_Modal = args.exchange_Modal)
    slices_train = torch.utils.data.ConcatDataset(volumes_train)
    slices_val = torch.utils.data.ConcatDataset(volumes_val)
    if args.prefetch:
        # load all data to RAM
        slices_train = Prefetch(slices_train)
        slices_val = Prefetch(slices_val)
    loader_train = torch.utils.data.DataLoader( \
            slices_train, batch_size=batchsize_train, shuffle=True, \
            num_workers=args.num_workers, pin_memory=True, drop_last=True)
    loader_val = torch.utils.data.DataLoader( \
            slices_val, batch_size=args.batch_size, shuffle=True, \
            num_workers=args.num_workers, pin_memory=True, drop_last=True)
    len_vis = 16
    col_vis = 4

    batch_vis = next(iter(torch.utils.data.DataLoader( \
            slices_val, batch_size=len_vis, shuffle=True)))
    batch_vis = [x.to(device, non_blocking=True) for x in batch_vis]
    batch_vis = [utils.complex_to_chan_dim(x) for x in batch_vis]    

    torch.manual_seed(int(time.time()))
    np.random.seed(int(time.time()))
    print('done, ' \
            + str(len(slices_train)) + ' / ' \
            + str(len(volumes_train)) + ' for training, ' \
            + str(len(slices_val)) + ' / ' \
            + str(len(volumes_val)) + ' for validation')

    # training.
    print('training...')
    last_loss, last_ckpt, last_disp = 0, 0, 0
    time_data, time_vis = 0, 0
    signal_earlystop = False
    iter_best = iter_cnt
    Eval_PSNR_best = None
    Eval_SSIM_best = None
    scalar_span = 50
    num_loss = cfg.num_recurrent+4 # loss_fidelity, loss_consistency, loss_VGG, loss_overall

    optim_R = torch.optim.AdamW(net.parameters(), \
            lr=cfg.lr, weight_decay=0)
    
    #scheduler = torch.optim.lr_scheduler.StepLR(optim_R,step_size=1,gamma=0.1)
    
    scalar = torch.cuda.amp.GradScaler(enabled=cfg.use_amp)
    
    record_train = np.zeros([len(loader_train)*args.epoch//scalar_span, num_loss]) # record train loss
    record_val = np.zeros([args.epoch, num_loss]) # record validation loss
    record_train_index = 0
    Eval_PSNR, Eval_SSIM = [], []
    time_start = time.time()
    for index_epoch in tqdm.trange(epoch_start, args.epoch, desc='epoch', leave=True):
        ###################  training ########################
        tqdm_iter = tqdm.tqdm(loader_train, desc='iter', \
                bar_format=str(batchsize_train)+': {n_fmt}/{total_fmt}'+\
                '[{elapsed}<{remaining},{rate_fmt}]'+'{postfix}', leave=False)
        
        #### learning rate decay ####
        if index_epoch%50==0:
            for param_group in optim_R.param_groups:
                param_group['lr'] = param_group['lr']*(0.5**(index_epoch//50))
                
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
                local_fidelities, loss_fidelity, loss_consistency, loss_VGG, loss_all = net(*batch)
            
            optim_R.zero_grad()
            scalar.scale(loss_all.mean()).backward()
            scalar.step(optim_R)
            scalar.update()
            
            del batch

            time_start = time.time()
            if iter_cnt % scalar_span == 0:
                last_loss = iter_cnt
                if cfg.GPUs>1: vis = net.module.get_vis('scalars') 
                else: vis = net.get_vis('scalars')
                for name, val in vis['scalars'].items():
                    writer.add_scalar('train/'+name, val, iter_cnt)
                if cfg.GPUs>1: vis = net.module.get_vis('histograms')
                else:vis = net.get_vis('histograms')
                for name, val in vis['histograms'].items():
                    writer.add_histogram( \
                            tag='train/'+name, \
                            global_step=iter_cnt, **val)
                del vis, name, val
                
                if cfg.GPUs>1:
                    record_train[record_train_index, :args.num_recurrent] = [(sum(i)/len(local_fidelities[0])).cpu() for i in local_fidelities]
                    record_train[record_train_index, args.num_recurrent:] = loss_fidelity.mean().cpu(), \
                                                loss_consistency.mean().cpu(), loss_VGG.mean(), loss_all.mean().cpu()

                else:
                    record_train[record_train_index, :args.num_recurrent] = local_fidelities
                    record_train[record_train_index, args.num_recurrent:] = loss_fidelity, loss_consistency, loss_VGG, loss_all

                np.save(args.logdir+'/train_loss_recorded', record_train)
                record_train_index += 1
                
            if iter_cnt % 5000 == 0:
                last_disp = iter_cnt
                net.eval()
                with torch.no_grad():
                    if cfg.GPUs>1:
                        net.module.test(*batch_vis)
                        vis = net.module.get_vis('images')
                    else: 
                        net.test(*batch_vis)
                        vis = net.get_vis('images')   
                for name, val in vis['images'].items():
                    torchvision.utils.save_image(val, \
                        args.logdir+'/res/'+'%010d_'%iter_cnt+name+'.jpg', \
                        nrow=len_vis//col_vis, padding=10, \
                        range=(0, 1), pad_value=0.5)
                del vis, name, val
            if (iter_cnt % 30000 == 0):
                last_ckpt = iter_cnt
                if cfg.GPUs>1: 
                    torch.save({'state_dict': net.module.state_dict(),
                                'config': cfg,
                                'epoch': index_epoch},
                               args.logdir+'/ckpt/ckpt_%010d.pt'%iter_cnt)
                else: 
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
                if cfg.GPUs>1: 
                    net.module.test(*batch)
                    stat_loss.append(net.module.Eval)
                    vis = net.module.get_vis('scalars')
                else:
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
            
            if cfg.GPUs>1: 
                record_val[index_epoch, :args.num_recurrent] = net.module.local_fidelities
                record_val[index_epoch, args.num_recurrent:] = net.module.loss_fidelity, \
                net.module.loss_consistency, net.module.loss_VGG, net.module.loss_all
            else:
                record_val[index_epoch, :args.num_recurrent] = net.local_fidelities
                record_val[index_epoch, args.num_recurrent:] = net.loss_fidelity, \
                net.loss_consistency, net.loss_VGG, net.loss_all    
 
            np.save(args.logdir+'/val_loss_recorded', record_val)
            np.save(args.logdir+'/PSNR', np.array(Eval_PSNR))
            np.save(args.logdir+'/SSIM', np.array(Eval_SSIM))

            
            if index_epoch%5==0:
                if cfg.GPUs>1:               
                    B, C, H, W = net.module.sens_maps.shape
                    for i in range(len(net.module.SMs)):
                        np.save(args.logdir+'/res/'+'%010d_'%iter_cnt+'_sensitivitymap'+str(i), \
                                utils.rss(net.module.SMs[i].view(-1, 1, H , W)).cpu().detach().numpy())
                            
                    torchvision.utils.save_image(utils.rss(net.module.sens_maps.view(-1, 1, H , W)), \
                                                 args.logdir+'/res/'+'%010d_'%iter_cnt+'_sensitivitymap.jpg', \
                                                 nrow=6, padding=10, range=(0, 1), pad_value=0.5)     
                    np.save(args.logdir+'/res/'+'%010d_'%iter_cnt+'_mask', net.module.mask.cpu().numpy()) # save mask 
                else: 
                    B, C, H, W = net.sens_maps.shape
                    for i in range(len(net.SMs)):
                        np.save(args.logdir+'/res/'+'%010d_'%iter_cnt+'_sensitivitymap'+str(i), \
                                utils.rss(net.SMs[i].view(-1, 1, H , W)).cpu().detach().numpy())
                            
                    torchvision.utils.save_image(utils.rss(net.sens_maps.view(-1, 1, H , W)), \
                                                 args.logdir+'/res/'+'%010d_'%iter_cnt+'_sensitivitymap.jpg', \
                                                 nrow=6, padding=10, range=(0, 1), pad_value=0.5)     
                    np.save(args.logdir+'/res/'+'%010d_'%iter_cnt+'_mask', net.mask.cpu().numpy()) # save mask 
                    
            

            # early_stop is enabled
            if (Eval_PSNR_best is None) or ((Eval_PSNR_current > Eval_PSNR_best) & (Eval_SSIM_current > Eval_SSIM_best)):
                Eval_PSNR_best = Eval_PSNR_current
                Eval_SSIM_best = Eval_SSIM_current
                iter_best = iter_cnt
                print('Current best iteration %d/%d:'%(iter_best, len(loader_train)*args.epoch), f' PSNR: {Eval_PSNR_best:.2f}', f', SSIM: {Eval_SSIM_best:.4f}')
                if cfg.GPUs>1: 
                    torch.save({'state_dict': net.module.state_dict(),
                                'config': cfg,
                                'epoch': index_epoch},
                                 args.logdir+'/ckpt/best.pt')
                else: 
                    torch.save({'state_dict': net.state_dict(),
                                'config': cfg,
                                'epoch': index_epoch},
                                 args.logdir+'/ckpt/best.pt')

            else:
                if iter_cnt >= args.early_stop + iter_best:
                    signal_earlystop=True
                    print('signal_earlystop set due to early_stop')

            #print('Current iteration %d/%d'%(iter_cnt, len(loader_train)*args.epoch), f' PSNR: {Eval_PSNR_current:.2f}', f', SSIM: {Eval_SSIM_current:.4f}', ', best iteration %d:'%(iter_best), f' PSNR: {Eval_PSNR_best:.2f}', f', SSIM: {Eval_SSIM_best:.4f}')
                
                      
    print('reached end of training loop, and signal_earlystop is '+str(signal_earlystop))
    writer.flush()
    writer.close()
    
    if cfg.GPUs>1: 
        torch.save({'state_dict': net.module.state_dict(),
                    'config': cfg,
                    'epoch': index_epoch},
                     args.logdir+'/ckpt/ckpt_%010d.pt'%iter_cnt)
    else: 
        torch.save({'state_dict': net.state_dict(),
                    'config': cfg,
                    'epoch': index_epoch},
                    args.logdir+'/ckpt/ckpt_%010d.pt'%iter_cnt)
    print('saved final ckpt:', args.logdir+'/ckpt/ckpt_%010d.pt'%iter_cnt)



if __name__ == '__main__':
    import argparse
    from autoGPU import autoGPU

    def try_int(v):
        # convert string to int
        try:
            v = int(v)
        except ValueError:
            v = int(float(v))
        assert v >= 0
        return v

    parser = argparse.ArgumentParser(description='jCAN')
    parser.add_argument('--logdir', metavar='logdir', \
                        type=str, default='/xxx',\
                        help='Save directory')
    parser.add_argument('--epoch', type=int, default=300, \
                        help='Epochs to train')
    parser.add_argument('--batch_size', type=int, default=1, \
                        help='Mini-batch size for training')
    parser.add_argument('--num_workers', type=int, default=os.cpu_count(), \
                        help='Number of threads for parallel preprocessing')
    parser.add_argument('--lr', type=float, default=2e-4, \
                        help='Learning rate')
    parser.add_argument('--early_stop', type=try_int, default=1000000, metavar='N', \
                        help='Stop training after val loss not going down for N iters')
    parser.add_argument('--n_SC', type=int, default=1, \
                        help='Number of cascades for SA and CA attention')
    parser.add_argument('--vgg_lambdas', type=float, default=2e-2, \
                        help='Weight of the VGG Loss')
    parser.add_argument('--vgg_indices', type=tuple, default=tuple([16]), \
                        help='Indices of the VGG layer as loss')
    parser.add_argument('--patch_size', type=tuple, default=tuple([1,1,1,1,1]), \
                        help='Patch size in ViT')   
    parser.add_argument('--beta', type=float, default=10, \
                        help='Weight of the consistency loss in K-space')
    parser.add_argument('--gamma', type=float, default=1, \
                        help='Weight of the gradient loss')
    parser.add_argument('--alpha', type=float, default=5, \
                        help='Weight of the k-space loss')
    parser.add_argument('--embed_dim', type=int, default=32, \
                        help='Dimension of embeddings in ViT')	
    parser.add_argument('--shape', type=int, default=320, \
                        help='Image shape, default 320')					
    parser.add_argument('--num_heads', type=int, default=8, \
                        help='Number of multiheads in ViT')
    parser.add_argument('--window_size', type=int, default=8, \
                        help='Window size of SwinTransformer')
    parser.add_argument('--num_recurrent', type=int, default=15, \
                        help='Number of DCRB')
    parser.add_argument('--mlp_ratio', type=int, default=16, \
                        help='Ratio in MLP')
    parser.add_argument('--sens_chans', type=int, default=8, \
                        help='Number of channels in sensitivity network')
    parser.add_argument('--sens_steps', type=int, default=4, \
                        help='Number of steps in initial sensitivity network')
    parser.add_argument('--GT', type=bool, default=True, \
                        help='if there is GT, default is True') 
    parser.add_argument('--exchange_Modal', type=bool, default=False, \
                        help='Augment data by exchanging the order of protocals, default is False') 
    parser.add_argument('--ds_ref', type=bool, default=False, \
                        help='Use gradient reference image as additional input, default is False') 
    parser.add_argument('--SR_scale', type=int, default=1, \
                        help='resoluton enhancement factor')
    parser.add_argument('--is_Kspace', type=bool, default=False, \
                        help='Raw data is in Kspace, default is False') 
    parser.add_argument('--is_Unet', type=bool, default=True, \
                        help='Use Unet for estimation of senstivity map, alternative is ResNet') 
    parser.add_argument('--mask', metavar='type', \
                        choices=['equispaced', 'random'], \
                        type=str, default = 'equispaced', help='types of mask')
    parser.add_argument('--sparsity', metavar='0-1', \
                        type=float, default=0.25, help='desired overall sparisity of masks without sparsity, 0.25 for 4X, 0.125 for 8X.')
    parser.add_argument('--train', type=str, default='/xxx.csv', \
                        help='path to csv file of training data')
    parser.add_argument('--val', default='/xxx.csv', \
                        type=str, help='path to csv file of validation data')
    parser.add_argument('--basepath', default='/xxx', \
                            type=str, help='Directory of data files')
    parser.add_argument('--coils', type=int, default=24, \
                        help='Number of coils')
    parser.add_argument('--protocals', metavar='NAME', \
                        type=str, default= ['T2','T1Flair'], nargs='*', help='Input modalities, first element is target, second is reference')
    parser.add_argument('--aux_aug', type=str, default = 'Rigid', \
                        choices=augment_funcs.keys(), help='Data augmentation')
    parser.add_argument('--prefetch', action='store_true')
    parser.add_argument('--use_amp', action='store_true')
    parser.add_argument('--force_gpu', action='store_true')
    args = parser.parse_args()

    if not args.force_gpu:
        autoGPU()

    main(args)
