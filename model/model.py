import logging
from collections import OrderedDict
import torch
import torch.nn as nn
import os
import model.networks as networks
from .base_model import BaseModel
from torch.nn.parallel import DistributedDataParallel as DDP

logger = logging.getLogger('base')


class DDPM(BaseModel):
    def __init__(self, opt):
        super(DDPM, self).__init__(opt)

        if opt['dist']:
            self.local_rank = torch.distributed.get_rank()
            torch.cuda.set_device(self.local_rank)
            device = torch.device("cuda", self.local_rank)
        # define network and load pretrained models
        self.netG = self.set_device(networks.define_G(opt))
        if opt['dist']:
            self.netG.to(device)
       
        # self.netG.to(device)
        if not opt['uncertainty_train']:
            self.netGU = self.set_device(networks.define_G(opt)) # uncertainty model
            if opt['dist']:
                self.netGU.to(device)
       

        self.schedule_phase = None
        self.opt = opt

        # set loss and load resume state
        self.set_loss()

        self.set_new_noise_schedule(
            opt['model']['beta_schedule']['train'], schedule_phase='train')
        if self.opt['phase'] == 'train':
            self.netG.train()
            # find the parameters to optimize
            if opt['model']['finetune_norm']:
                optim_params = []
                for k, v in self.netG.named_parameters():
                    v.requires_grad = False
                    if k.find('transformer') >= 0:
                        v.requires_grad = True
                        v.data.zero_()
                        optim_params.append(v)
                        logger.info(
                            'Params [{:s}] initialized to 0 and will optimize.'.format(k))
            else:
                optim_params = list(self.netG.parameters())

            self.optG = torch.optim.Adam(
                optim_params, lr=opt['train']["optimizer"]["lr"])
            self.log_dict = OrderedDict()

        if not opt['uncertainty_train'] and self.opt['phase'] == 'train':
            self.netGU.load_state_dict(torch.load(self.opt['path']['resume_state']+'_gen.pth'), strict=True) # use uncertainty model for initialization 
            if opt['dist']:
                self.netGU = DDP(self.netGU, device_ids=[self.local_rank], output_device=self.local_rank, find_unused_parameters=True)

        if self.opt['phase'] == 'test':
            try:
                self.netG.load_state_dict(torch.load(self.opt['path']['resume_state']), strict=True)
            except Exception:
                self.netG = nn.DataParallel(self.netGU)
                self.netG.load_state_dict(torch.load(self.opt['path']['resume_state']), strict=True)
        else:
            self.load_network()
            if opt['dist']:
                self.netG = DDP(self.netG, device_ids=[self.local_rank], output_device=self.local_rank,find_unused_parameters=True)
        self.print_network()

    def feed_data(self, data):

        dic = {}

        if self.opt['dist']:
            dic = {}
            dic['LQ'] = data['LQ'].to(self.local_rank)
            dic['GT'] = data['GT'].to(self.local_rank)
            self.data = dic
        else:
            dic['LQ'] = data['LQ']
            dic['GT'] = data['GT']

            self.data = self.set_device(dic)

    def optimize_parameters(self):

        self.optG.zero_grad()

        if not self.opt['uncertainty_train']:
            if self.opt['dist']:
                l_pix, l_gsad = self.netG(self.data, self.netGU.module.denoise_fn)
            else:
                l_pix, l_gsad = self.netG(self.data, self.netGU.denoise_fn)

            # need to average in multi-gpu
            b, c, h, w = self.data['LQ'].shape

            num_clusters = 6
            l_pix = l_pix.sum()/int(b*c*h*w)
            l_gsad = l_gsad.sum()/int(b*num_clusters)
            # l_svd = l_svd.sum()/int(b*c*h*w)
            loss = l_pix + l_gsad
            loss.backward()
            self.optG.step()

            # set log
            self.log_dict['total_loss'] = loss.item()
            self.log_dict['l_1'] = l_pix.item()
            self.log_dict['l_gsad'] = l_gsad.item()
        else:
            l_pix = self.netG(self.data)

            b, c, h, w = self.data['LQ'].shape

            l_pix = l_pix.sum()/int(b*c*h*w)
            l_pix.backward()
            self.optG.step()

            # set log
            self.log_dict['l_u'] = l_pix.item()


    def test(self, continous=False):
        self.netG.eval()
        with torch.no_grad():
            if isinstance(self.netG, nn.DataParallel):
                self.SR = self.netG.module.super_resolution(
                    self.data['LQ'], continous)
                
            else:
                if self.opt['dist']:
                    self.SR = self.netG.module.super_resolution(self.data['LQ'], continous)
                else:
                    self.SR = self.netG.super_resolution(self.data['LQ'], continous)

        self.netG.train()

    def sample(self, batch_size=1, continous=False):
        self.netG.eval()
        with torch.no_grad():
            if isinstance(self.netG, nn.DataParallel):
                self.SR = self.netG.module.sample(batch_size, continous)
            else:
                self.SR = self.netG.sample(batch_size, continous)
        self.netG.train()

    def set_loss(self):
        if isinstance(self.netG, nn.DataParallel):
            self.netG.module.set_loss(self.device)
        else:
            self.netG.set_loss(self.device)

        if not self.opt['uncertainty_train']:
            if isinstance(self.netGU, nn.DataParallel):
                self.netGU.module.set_loss(self.device)
            else:
                self.netGU.set_loss(self.device)

    def set_new_noise_schedule(self, schedule_opt, schedule_phase='train'):

        if self.opt['dist']:
            # local_rank = torch.distributed.get_rank()
            device = torch.device("cuda", self.local_rank)
            if self.schedule_phase is None or self.schedule_phase != schedule_phase:
                self.schedule_phase = schedule_phase
                if isinstance(self.netG, nn.DataParallel):
                    self.netG.module.set_new_noise_schedule(
                        schedule_opt, self.device)
                else:
                    self.netG.set_new_noise_schedule(schedule_opt, device)

                if not self.opt['uncertainty_train']:
                    if isinstance(self.netGU, nn.DataParallel):
                        self.netGU.module.set_new_noise_schedule(
                            schedule_opt, self.device)
                    else:
                        self.netGU.set_new_noise_schedule(schedule_opt, device)
        else:
            self.schedule_phase = schedule_phase
            if isinstance(self.netG, nn.DataParallel):
                self.netG.module.set_new_noise_schedule(
                    schedule_opt, self.device)
            else:
                self.netG.set_new_noise_schedule(schedule_opt, self.device)

            if not self.opt['uncertainty_train']:
                if isinstance(self.netGU, nn.DataParallel):
                    self.netGU.module.set_new_noise_schedule(
                        schedule_opt, self.device)
                else:
                    self.netGU.set_new_noise_schedule(schedule_opt, self.device)

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, need_LR=True, sample=False):
        out_dict = OrderedDict()
        if sample:
            out_dict['SAM'] = self.SR.detach().float().cpu()
        else:
            out_dict['HQ'] = self.SR.detach().float().cpu()
            out_dict['INF'] = self.data['LQ'].detach().float().cpu()
            out_dict['GT'] = self.data['GT'].detach()[0].float().cpu()
            if need_LR and 'LR' in self.data:
                out_dict['LQ'] = self.data['LQ'].detach().float().cpu()
            else:
                out_dict['LQ'] = out_dict['INF']
        return out_dict

    def print_network(self):
        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, nn.DataParallel):
            net_struc_str = '{} - {}'.format(self.netG.__class__.__name__,
                                             self.netG.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netG.__class__.__name__)

        logger.info(s)

        logger.info(
            'Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))

    def save_network(self, epoch, iter_step):
        gen_path = os.path.join(
            self.opt['path']['checkpoint'], 'I{}_E{}_gen.pth'.format(iter_step, epoch))
        opt_path = os.path.join(
            self.opt['path']['checkpoint'], 'I{}_E{}_opt.pth'.format(iter_step, epoch))
        
        # gen
        network = self.netG
        if isinstance(self.netG, nn.DataParallel):
            network = network.module
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, gen_path)
        # opt
        opt_state = {'epoch': epoch, 'iter': iter_step,
                     'scheduler': None, 'optimizer': None}
        opt_state['optimizer'] = self.optG.state_dict()
        torch.save(opt_state, opt_path)

        if self.opt['uncertainty_train']:
            uncertainty_save_dir = './checkpoints/uncertainty/'
            os.makedirs(uncertainty_save_dir, exist_ok=True)
            ut_gen_path = os.path.join(
                './checkpoints/uncertainty/', 'latest_gen.pth'.format(iter_step, epoch))
            ut_opt_path = os.path.join(
                './checkpoints/uncertainty/', 'latest_opt.pth'.format(iter_step, epoch))
            torch.save(state_dict, ut_gen_path)
            torch.save(opt_state, ut_opt_path)

        logger.info(
            'Saved model in [{:s}] ...'.format(gen_path))

    def load_network(self):
        load_path = self.opt['path']['resume_state']
        if load_path is not None:
            logger.info(
                'Loading pretrained model for G [{:s}] ...'.format(load_path))
            gen_path = '{}_gen.pth'.format(load_path)
            opt_path = '{}_opt.pth'.format(load_path)
            # gen
            network = self.netG
            if isinstance(self.netG, nn.DataParallel):
                network = network.module

            # network = nn.DataParallel(network).cuda()

            network.load_state_dict(torch.load(
                gen_path), strict=(not self.opt['model']['finetune_norm']))
            # network.load_state_dict(torch.load(
            #     gen_path), strict=False)
            if self.opt['phase'] == 'train':
                # optimizer
                # opt = torch.load(opt_path)
                # self.optG.load_state_dict(opt['optimizer'])
                # self.begin_step = opt['iter']
                # self.begin_epoch = opt['epoch']
                self.begin_step = 0
                self.begin_epoch = 0
