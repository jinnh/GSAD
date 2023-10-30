import math
import torch
from torch import device, nn, einsum
import torch.nn.functional as F
from inspect import isfunction
from functools import partial
import numpy as np
from tqdm import tqdm
import cv2
from .kmean_torch import kmeans_core
from kmeans_pytorch import kmeans
from torchvision import transforms
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import MeanShift
from sklearn.cluster import DBSCAN
from sklearn.cluster import SpectralClustering

transform = transforms.Lambda(lambda t: (t + 1) / 2)

def _warmup_beta(linear_start, linear_end, n_timestep, warmup_frac):
    betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    warmup_time = int(n_timestep * warmup_frac)
    betas[:warmup_time] = np.linspace(
        linear_start, linear_end, warmup_time, dtype=np.float64)
    return betas


def make_beta_schedule(schedule, n_timestep, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
    if schedule == 'quad':
        betas = np.linspace(linear_start ** 0.5, linear_end ** 0.5,
                            n_timestep, dtype=np.float64) ** 2
    elif schedule == 'linear':
        betas = np.linspace(linear_start, linear_end,
                            n_timestep, dtype=np.float64)
    elif schedule == 'warmup10':
        betas = _warmup_beta(linear_start, linear_end,
                             n_timestep, 0.1)
    elif schedule == 'warmup50':
        betas = _warmup_beta(linear_start, linear_end,
                             n_timestep, 0.5)
    elif schedule == 'const':
        betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    elif schedule == 'jsd':  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1. / np.linspace(n_timestep,
                                 1, n_timestep, dtype=np.float64)
    elif schedule == "cosine":
        timesteps = (
            torch.arange(n_timestep + 1, dtype=torch.float64) /
            n_timestep + cosine_s
        )
        alphas = timesteps / (1 + cosine_s) * math.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = betas.clamp(max=0.999)
    else:
        raise NotImplementedError(schedule)
    return betas


# gaussian diffusion trainer class

def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        denoise_fn,
        image_size,
        channels=3,
        loss_type='l1',
        conditional=True,
        schedule_opt=None
    ):
        super().__init__()
        self.channels = channels
        self.image_size = image_size
        self.denoise_fn = denoise_fn
        self.loss_type = loss_type
        self.conditional = conditional
        if schedule_opt is not None:
            pass
            # self.set_new_noise_schedule(schedule_opt)

    def set_loss(self, device):
        if self.loss_type == 'l1':
            self.loss_func = nn.L1Loss(reduction='sum').to(device)
        elif self.loss_type == 'l2':
            self.loss_func = nn.MSELoss(reduction='sum').to(device)
        else:
            raise NotImplementedError()

    def set_new_noise_schedule(self, schedule_opt, device):
        to_torch = partial(torch.tensor, dtype=torch.float32, device=device)

        betas = make_beta_schedule(
            schedule=schedule_opt['schedule'],
            n_timestep=schedule_opt['n_timestep'],
            linear_start=schedule_opt['linear_start'],
            linear_end=schedule_opt['linear_end'])
        betas = betas.detach().cpu().numpy() if isinstance(
            betas, torch.Tensor) else betas
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])
        self.sqrt_alphas_cumprod_prev = np.sqrt(
            np.append(1., alphas_cumprod))

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev',
                             to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod',
                             to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod',
                             to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod',
                             to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod',
                             to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod',
                             to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * \
            (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance',
                             to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(
            np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

    def predict_start_from_noise(self, x_t, t, noise):
        return self.sqrt_recip_alphas_cumprod[t] * x_t - \
            self.sqrt_recipm1_alphas_cumprod[t] * noise

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = self.posterior_mean_coef1[t] * \
            x_start + self.posterior_mean_coef2[t] * x_t
        posterior_log_variance_clipped = self.posterior_log_variance_clipped[t]
        return posterior_mean, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, clip_denoised: bool, condition_x=None):
        batch_size = x.shape[0]
        noise_level = torch.FloatTensor(
            [self.sqrt_alphas_cumprod_prev[t+1]]).repeat(batch_size, 1).to(x.device)
        if condition_x is not None:
            x_recon = self.predict_start_from_noise(
                x, t=t, noise=self.denoise_fn(torch.cat([condition_x, x], dim=1), noise_level)[0])
        else:
            x_recon = self.predict_start_from_noise(
                x, t=t, noise=self.denoise_fn(x, noise_level))

        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        model_mean, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, t, clip_denoised=True, condition_x=None):
        model_mean, model_log_variance = self.p_mean_variance(
            x=x, t=t, clip_denoised=clip_denoised, condition_x=condition_x)
        noise = torch.randn_like(x) if t > 0 else torch.zeros_like(x)
        # print(model_log_variance)
        # print(model_log_variance.shape)
        return model_mean + noise * (0.5 * model_log_variance).exp()

    @torch.no_grad()
    def p_sample_loop(self, x_in, continous=False):
        device = self.betas.device
        sample_inter = (1 | (self.num_timesteps//10))
        if not self.conditional:
            shape = x_in
            img = torch.randn(shape, device=device)
            ret_img = img
            for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
                img = self.p_sample(img, i)
                if i % sample_inter == 0:
                    ret_img = torch.cat([ret_img, img], dim=0)
        else:
            x = x_in
            shape = x.shape
            img = torch.randn(shape, device=device)
            ret_img = x
            for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
                img = self.p_sample(img, i, condition_x=x)
                if i % sample_inter == 0:
                    ret_img = torch.cat([ret_img, img], dim=0)
        if continous:
            return ret_img
        else:
            return ret_img[-1]

    @torch.no_grad()
    def sample(self, batch_size=1, continous=False):
        image_size = self.image_size
        channels = self.channels
        return self.p_sample_loop((batch_size, channels, image_size, image_size), continous)

    @torch.no_grad()
    def super_resolution(self, x_in, continous=False):
        return self.p_sample_loop(x_in, continous)

    def q_sample(self, x_start, continuous_sqrt_alpha_cumprod, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        return (
            continuous_sqrt_alpha_cumprod * x_start +
            (1 - continuous_sqrt_alpha_cumprod**2).sqrt() * noise
        )


    def draw_features(self, x, savename):
        img = x[0, 0, :, :]
        pmin = np.min(img)
        pmax = np.max(img)
        img = ((img - pmin) / (pmax - pmin + 0.000001)) * 255  
        img = img.astype(np.uint8)  
        img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
        cv2.imwrite(savename,img)

    def predict_start(self, x_t, continuous_sqrt_alpha_cumprod, noise):
        return (1. / continuous_sqrt_alpha_cumprod) * x_t - \
            (1. / continuous_sqrt_alpha_cumprod**2 - 1).sqrt() * noise

    def predict_t_minus1(self, x, t, continuous_sqrt_alpha_cumprod, noise, clip_denoised=True):

        x_recon = self.predict_start(x, 
                    continuous_sqrt_alpha_cumprod=continuous_sqrt_alpha_cumprod.view(-1, 1, 1, 1), 
                    noise=noise)

        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        model_mean, model_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)

        noise_z = torch.randn_like(x) if t > 0 else torch.zeros_like(x)

        return model_mean + noise_z * (0.5 * model_log_variance).exp()       


    def to_patches(sefl, data, kernel_size):

        patches = nn.Unfold(kernel_size=kernel_size, stride=kernel_size)(torch.mean(data, dim=1, keepdim=True))
        patches = patches.transpose(2,1)

        return patches


    def calcu_kmeans(self, data, num_clusters):

        [b, h, w] = data.shape
        cluster_ids_all = np.empty([b, h])
        cluster_ids_all = torch.from_numpy(cluster_ids_all)
        for i in range(b):
            # cluster_ids, cluster_centers = kmeans(
            #     X=data[i,:,:], num_clusters=num_clusters, distance='euclidean', device=torch.device('cuda:0')
            # )

            # DBSCAN
            # model = DBSCAN(eps=5)
            # cluster_ids = model.fit_predict(data[i,:,:].cpu())
            # cluster_ids = torch.from_numpy(cluster_ids).cuda()

            # # MeanShift
            # model = MeanShift()
            # cluster_ids = model.fit_predict(data[i,:,:].cpu())
            # cluster_ids = torch.from_numpy(cluster_ids).cuda()

            # # Spectral Clustering
            # model = SpectralClustering(n_clusters=num_clusters)
            # cluster_ids = model.fit_predict(transform(data[i,:,:].cpu()))
            # cluster_ids = torch.from_numpy(cluster_ids).cuda()

            # Hierarchical Clustering
            model = AgglomerativeClustering(n_clusters=num_clusters)
            cluster_ids = model.fit_predict(transform(data[i,:,:].cpu()))
            cluster_ids = torch.from_numpy(cluster_ids).cuda()

            # # gmm
            # model = GaussianMixture(n_components=num_clusters)
            # model.fit(data[i,:,:].cpu())
            # cluster_ids = model.predict(data[i,:,:].cpu())
            # cluster_ids = torch.from_numpy(cluster_ids).cuda()
            # print(cluster_ids)

            # # kmeans
            # km = kmeans_core(k=num_clusters,data_array=data[i,:,:].cpu().numpy(),batch_size=400,epochs=1000)
            # km.run()
            # cluster_ids = km.idx

            # print(cluster_ids)
            cluster_ids_all[i, :] = cluster_ids
        
        return cluster_ids_all

    def calcu_svd(self, data):

        u, sv, v = torch.svd(data)
        #sv_F2 = torch.norm(sv, dim=1)
        #sv_F2 = sv_F2.unsqueeze(1)
        #normalized_sv = sv / sv_F2

        return sv

    def calcu_svd_distance(self, data1, data2, cluster_ids, num_clusters):

        [b, h, w] = data1.shape 
        sv_ab_dis = np.empty([b, num_clusters])
        sv_ab_dis = torch.from_numpy(sv_ab_dis)
        for i in range(num_clusters):

            indices = (cluster_ids[0] ==i).nonzero(as_tuple=True)[0]
            
            if len(indices)==0:
                sv_ab_dis[:, i] = 1e-5
            else:
                data1_select = torch.index_select(data1, 1, indices.cuda())
                data2_select = torch.index_select(data2, 1, indices.cuda())
                sv1 = self.calcu_svd(data1_select.cpu())
                sv2 = self.calcu_svd(data2_select.cpu())
   
                sv_ab_dis_i = torch.abs(sv1 - sv2)
                sv_ab_dis[:, i] = torch.sum(sv_ab_dis_i, dim=1)
        return sv_ab_dis

    def global_aware_losses(self, x_in, uct_model, noise=None):

        x_start = x_in['GT']
        [b, c, h, w] = x_start.shape

        t = np.random.randint(1, self.num_timesteps + 1)

        continuous_sqrt_alpha_cumprod = torch.FloatTensor(
            np.random.uniform(
                self.sqrt_alphas_cumprod_prev[t-1],
                self.sqrt_alphas_cumprod_prev[t],
                size=b
            )
        ).to(x_start.device)
        continuous_sqrt_alpha_cumprod = continuous_sqrt_alpha_cumprod.view(
            b, -1)

        noise = default(noise, lambda: torch.randn_like(x_start))

        x_noisy = self.q_sample(
            x_start=x_start, continuous_sqrt_alpha_cumprod=continuous_sqrt_alpha_cumprod.view(-1, 1, 1, 1), noise=noise)

        if not self.conditional:
            x_recon = self.denoise_fn(x_noisy, continuous_sqrt_alpha_cumprod)
        else:
            x_recon, _ = self.denoise_fn(
                torch.cat([x_in['LQ'], x_noisy], dim=1), continuous_sqrt_alpha_cumprod)
            with torch.no_grad():
                _, Pt = uct_model(
                    torch.cat([x_in['LQ'], x_noisy], dim=1), continuous_sqrt_alpha_cumprod)

        # global structure-aware constrain
        x_0 = x_start
        x_0_patches = self.to_patches(x_0, kernel_size=8) 

        x_t_1 = self.predict_t_minus1(x_noisy, t-1, 
                continuous_sqrt_alpha_cumprod=continuous_sqrt_alpha_cumprod.view(-1, 1, 1, 1), 
                noise=x_recon) 
        x_t_1_patches = self.to_patches(x_t_1, kernel_size=8)

        num_clusters = 6
        cluster_ids = self.calcu_kmeans(x_0_patches, num_clusters=num_clusters)
        svd_dis = self.calcu_svd_distance(x_0_patches, x_t_1_patches, cluster_ids=cluster_ids, num_clusters=num_clusters)

        b, c, h, w = Pt.shape
        s1 = Pt.view(b, c, -1)
        pmin = torch.min(s1, dim=-1)
        pmin = pmin[0].unsqueeze(dim=-1).unsqueeze(dim=-1)
        pmax = torch.max(s1, dim=-1)
        pmax = pmax[0].unsqueeze(dim=-1).unsqueeze(dim=-1)

        Pt = (Pt - pmin) / (pmax - pmin + 0.000001)
        # Pt = Pt * 0.5 + 0.5

        epsilon_pred = torch.mul(x_recon, Pt)
        epsilon = torch.mul(noise, Pt)
        lambda_ = 10
        loss_pix = self.loss_func(epsilon_pred, epsilon) * lambda_

        kappa = continuous_sqrt_alpha_cumprod**4
        loss_s = svd_dis.cuda() * kappa 

        # --------------------- perceptual_loss --------------------- #
        # x_0 = x_start
        # x_t_1 = self.predict_t_minus1(x_noisy, t-1, continuous_sqrt_alpha_cumprod=continuous_sqrt_alpha_cumprod.view(-1, 1, 1, 1), noise=x_recon) 
        
        # loss_pix = self.loss_func(x_recon, noise)
        # p_loss = self.perceptual_loss(transform(x_t_1.cpu()).cuda(), transform(x_0.cpu()).cuda(), vgg_model)
        # loss_s = p_loss # * continuous_sqrt_alpha_cumprod**4


        # -------------------- L1 ||x_t_1 - x0|| -------------------- #
        # loss_pix = self.loss_func(x_recon, noise) 
        # svd_dis = self.loss_func(x_t_1, x_0)
        # print(t, loss_pix, svd_dis)
        # loss_s = svd_dis #* continuous_sqrt_alpha_cumprod**4

        return loss_pix, loss_s

    def forward(self, x, uct_model, *args, **kwargs):
        return self.global_aware_losses(x, uct_model, *args, **kwargs)
