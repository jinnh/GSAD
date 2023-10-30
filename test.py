import os
from os.path import basename
import math
import argparse
import random
import logging
import cv2
import sys
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn

import options.options as option
from utils import util
from data import create_dataloader
from data.LoL_dataset import LOLv1_Dataset, LOLv2_Dataset
import torchvision.transforms as T
import lpips
import model as Model
import core.logger as Logger
import core.metrics as Metrics
from torchvision import transforms


transform = transforms.Lambda(lambda t: (t * 2) - 1)

def main():

    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, help='Path to option YMAL file.',
                            default='./config/LOLv1.yml') # 
    parser.add_argument('--launcher', choices=['none', 'pytorch'], default='none',
                        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--tfboard', action='store_true')


    parser.add_argument('-c', '--config', type=str, default='config/lolv1_test.json',
                        help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['train', 'val'],
                        help='Run either train(training) or val(generation)', default='train')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default="0")
    parser.add_argument('-debug', '-d', action='store_true')
    parser.add_argument('-log_eval', action='store_true')

    args = parser.parse_args()
    opt = Logger.parse(args)
    opt = Logger.dict_to_nonedict(opt)
    opt_dataset = option.parse(args.dataset, is_train=True)


    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    opt['phase'] = 'test'
    opt['uncertainty_train'] = False

    #### distributed training settings
    opt['dist'] = False
    rank = -1
    print('Disabled distributed training.')

    #### mkdir and loggers
    if rank <= 0:  # normal training (rank -1) OR distributed training (rank 0)
        # config loggers. Before it, the log will not work
        util.setup_logger('base', opt['path']['log'], 'train_' + opt['name'], level=logging.INFO,
                          screen=True, tofile=True)
        util.setup_logger('val', opt['path']['log'], 'val_' + opt['name'], level=logging.INFO,
                          screen=True, tofile=True)
        logger = logging.getLogger('base')
        logger.info(option.dict2str(opt))

    # convert to NoneDict, which returns None for missing keys
    opt = option.dict_to_nonedict(opt)

    #### seed
    seed = opt['seed']
    if seed is None:
        seed = random.randint(1, 10000)
    if rank <= 0:
        logger.info('Seed: {}'.format(seed))
    util.set_random_seed(seed)

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    #### create train and val dataloader
    if opt_dataset['dataset'] == 'LOLv1':
        dataset_cls = LOLv1_Dataset
    elif opt_dataset['dataset'] == 'LOLv2':
        dataset_cls = LOLv2_Dataset

    else:
        raise NotImplementedError()

    for phase, dataset_opt in opt_dataset['datasets'].items():
        if phase == 'val':
            val_set = dataset_cls(opt=dataset_opt, train=False, all_opt=opt_dataset)
            val_loader = create_dataloader(val_set, dataset_opt, opt_dataset, None)

    # model
    diffusion = Model.create_model(opt)
    logger.info('Initial Model Finished')

    diffusion.set_new_noise_schedule(
        opt['model']['beta_schedule']['train'], schedule_phase='train')


    loss_fn_vgg = lpips.LPIPS(net='alex') 
    result_path = '{}'.format(opt['path']['results'])
    result_path_gt = result_path+'/gt/'
    result_path_out = result_path+'/output/'
    result_path_input = result_path+'/input/'
    os.makedirs(result_path_gt, exist_ok=True)
    os.makedirs(result_path_out, exist_ok=True)
    os.makedirs(result_path_input, exist_ok=True)

    diffusion.set_new_noise_schedule(
        opt['model']['beta_schedule']['val'], schedule_phase='val')
    
    logger_val = logging.getLogger('val')  # validation logger

    avg_psnr = 0.0
    avg_ssim = 0.0
    idx = 0
    lpipss = []
    
    for val_data in val_loader:

        idx += 1
        diffusion.feed_data(val_data)
        diffusion.test(continous=False)

        visuals = diffusion.get_current_visuals()
        
        normal_img = Metrics.tensor2img(visuals['HQ'])
        if normal_img.shape[0] != normal_img.shape[1]: # lolv1 and lolv2-real
            normal_img = normal_img[8:408, 4:604,:]
        gt_img = Metrics.tensor2img(visuals['GT'])
        ll_img = Metrics.tensor2img(visuals['LQ'])

        img_mode = 'single'
        if img_mode == 'single':
            util.save_img(
                gt_img, '{}/{}_gt.png'.format(result_path_gt, idx))
            util.save_img(
                ll_img, '{}/{}_lq.png'.format(result_path_input, idx))
            # util.save_img(
            #     normal_img, '{}/{}_normal_noadjust.png'.format(result_path, idx))
        else:
            util.save_img(
                gt_img, '{}/{}_gt.png'.format(result_path, idx))
            util.save_img(
                normal_img, '{}/{}_{}_normal_process.png'.format(result_path, idx))
            # for i in range(visuals['HQ'].shape[0]):
            #     util.save_img(Metrics.tensor2img(visuals['HQ'][i]), '{}/{}_{}_normal.png'.format(result_path, idx, i))
            # util.save_img(
            #     Metrics.tensor2img(visuals['HQ'][-1]), '{}/{}_normal.png'.format(result_path, idx))
            normal_img = Metrics.tensor2img(visuals['HQ'][-1])

        # Similar to LLFlow, we follow a similar way of 'Kind' to finetune the overall brightness 
        # as illustrated in Line 73 (https://github.com/zhangyhuaee/KinD/blob/master/evaluate_LOLdataset.py).
        gt_img = gt_img / 255.
        normal_img = normal_img / 255.
        mean_gray_out = cv2.cvtColor(normal_img.astype(np.float32), cv2.COLOR_BGR2GRAY).mean()
        mean_gray_gt = cv2.cvtColor(gt_img.astype(np.float32), cv2.COLOR_BGR2GRAY).mean()
        normal_img_adjust = np.clip(normal_img * (mean_gray_gt / mean_gray_out), 0, 1)

        normal_img = (normal_img_adjust * 255).astype(np.uint8)
        gt_img = (gt_img * 255).astype(np.uint8)

        psnr = util.calculate_psnr(normal_img, gt_img)
        ssim = util.calculate_ssim(normal_img, gt_img)
        
        util.save_img(normal_img, '{}/{}_normal.png'.format(result_path_out, idx))

        # lpips
        img_hq = np.transpose(normal_img/255, (2, 0, 1))
        img_hq = transform(torch.from_numpy(img_hq).unsqueeze(0))
        img_gt = np.transpose(gt_img/255, (2, 0, 1))
        img_gt = transform(torch.from_numpy(img_gt).unsqueeze(0))
        lpips_ = loss_fn_vgg(img_hq.to(torch.float32), img_gt.to(torch.float32))
        
        # lpips_ = loss_fn_vgg(visuals['HQ'], visuals['GT'])
        lpipss.append(lpips_.detach().numpy())

        logger_val.info('### {} cPSNR: {:.4e} cSSIM: {:.4e} cLPIPS: {:.4e}'.format(idx, psnr, ssim, lpips_.detach().numpy()[0][0][0][0]))
        avg_ssim += ssim
        avg_psnr += psnr

    avg_psnr = avg_psnr / idx
    avg_ssim = avg_ssim / idx

    # log
    logger_val.info('# Validation # avgPSNR: {:.4e} avgSSIM: {:.4e} avgLPIPS: {:.4e}'.format(avg_psnr, avg_ssim, np.mean(lpipss)))



if __name__ == '__main__':
    main()
