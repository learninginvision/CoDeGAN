import torch
import torch.nn as nn
import torch.nn.functional as F
from   PIL.Image import NONE
from   utils.util import *
from   utils.sup_contrastive_loss import SupConLoss
from   model import Generator, Discriminator, Encoder


class Solver():
    def __init__(self, config):

        self.zn_dim = config['zn_dim']
        self.zc_dim = config['zc_dim']
        self.fz_dim = config['fz_dim']
        self.fc_dim = config['fc_dim']

        self.beta_zn = config['beta_zn']
        self.beta_zc = config['beta_zc']

        self.r_batch_size = config['r_batch_size']

        self.ch      = config['ch']
        self.im_size = config['im_size']

        self._init       = config['_init']
        self.temp_C      = config['temp_C']
        self.loss_metric = config['loss_metric']
        self.device = 'cuda:%d'%(config['gpu_id'])

        self.D = Discriminator(self.im_size, self.ch, self.loss_metric, self._init).to(self.device)
        self.E = Encoder(self.fz_dim, self.fc_dim, self.im_size, self.ch, self._init).to(self.device)
        self.G = Generator(self.zn_dim, self.zc_dim, self.im_size, self.ch, self._init).to(self.device)

        self.g_lr = config['g_lr']
        self.d_lr = config['d_lr']
        self.e_lr = config['e_lr']

        self.optimizer_G = torch.optim.Adam(self.G.parameters(), lr=self.g_lr, betas=(0.5, 0.999), eps=1e-8)
        self.optimizer_D = torch.optim.Adam(self.D.parameters(), lr=self.d_lr, betas=(0.5, 0.999), eps=1e-8)
        self.optimizer_E = torch.optim.Adam(self.E.parameters(), lr=self.e_lr, betas=(0.5, 0.999), eps=1e-8)

        self.bceloss = nn.BCELoss().to(self.device)
        self.mseloss = nn.MSELoss().to(self.device)
        self.C_loss = SupConLoss(temperature=self.temp_C, device=self.device).to(self.device)


    def forward_G(self, z, zc, batch_size, real_data):
        fake_q = self.G(z)
        d_fake = self.D(fake_q)

        # add contrastive
        fake_k = train_transform(fake_q)

        fz, fc_q = self.E(fake_q)
        _ , fc_k = self.E(fake_k)

        if real_data is not None:
            (real_q, real_label) = real_data.sample_paired_all_value_batch(self.r_batch_size, 0)
            real_k = train_transform(real_q)
            zc     = torch.cat([zc  , real_label.to(self.device)], dim=0)
            fc_q   = torch.cat([fc_q, self.E(real_q.to(self.device))[1]], dim=0)
            fc_k   = torch.cat([fc_k, self.E(real_k.to(self.device))[1]], dim=0)
            batch_size = zc.shape[0]

        fc = torch.cat([fc_q, fc_k], dim=0)
        f1, f2 = torch.split(fc, [batch_size, batch_size], dim=0)
        fc = torch.cat([f1.unsqueeze(1),f2.unsqueeze(1)],dim=1)

        return d_fake, fz, fc, zc
        
    def optimize_parametersG(self, batch_size, real_data=None):
        self.set_requires_grad(self.D, False)
        self.set_requires_grad(self.E, False)
        self.set_requires_grad(self.G, True )
        self.optimizer_G.zero_grad()

        z, zn, zc = sample_noise(batch_size, self.zn_dim, self.zc_dim, self.device)
        d_fake, fz, fc, zc = self.forward_G(z, zc, batch_size, real_data)
        zc_mask = get_mask_zc_discrete(batch_size, zc).to(self.device)

        if self.loss_metric   == 'wgan':
            gloss = torch.mean(d_fake)
        elif self.loss_metric == 'vanline':
            label_real = torch.full((batch_size, 1), 1, device=self.device)
            gloss = self.bceloss(d_fake, label_real.float())
        elif self.loss_metric == 'hinge':
            gloss = -d_fake.mean()

        ecloss = self.C_loss(fc, zc_mask)
        ezloss = self.mseloss(fz, zn)
        gloss  = gloss + self.beta_zc*ecloss + self.beta_zn*ezloss

        gloss.backward()
        self.optimizer_G.step()
        return gloss.item()
        

    def forward_E(self, z, zc, batch_size, real_data):
        fake_q = self.G(z)
        fake_k = train_transform(fake_q)

        # add contrastive
        fz, fc_q = self.E(fake_q)
        _ , fc_k = self.E(fake_k)

        if real_data is not None:
            (real_q, real_label) = real_data.sample_paired_all_value_batch(self.r_batch_size, 0)
            real_k = train_transform(real_q)
            zc     = torch.cat([zc  , real_label.to(self.device)], dim=0)
            fc_q   = torch.cat([fc_q, self.E(real_q.to(self.device))[1]], dim=0)
            fc_k   = torch.cat([fc_k, self.E(real_k.to(self.device))[1]], dim=0)
            batch_size = zc.shape[0]

        zc_mask = get_mask_zc_discrete(batch_size, zc).to(self.device)
        fc      = torch.cat([fc_q, fc_k], dim=0)
        f1, f2  = torch.split(fc, [batch_size, batch_size], dim=0)
        fc      = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)

        return fz, fc, zc_mask

    def optimize_parametersE(self, batch_size, real_data=None):
        self.set_requires_grad(self.D, False)
        self.set_requires_grad(self.E, True )
        self.set_requires_grad(self.G, False)
        self.optimizer_E.zero_grad()

        z , zn, zc = sample_noise(batch_size, self.zn_dim, self.zc_dim, self.device)
        fz, fc, zc_mask = self.forward_E(z, zc, batch_size, real_data)

        ecloss = self.C_loss (fc, zc_mask)
        ezloss = self.mseloss(fz, zn)
        eloss  = self.beta_zc*ecloss + self.beta_zn*ezloss

        eloss.backward()
        self.optimizer_E.step()
        return ecloss.item(), ezloss.item()


    def forward_D(self, real, fake):
        d_real = self.D(real)
        d_fake_detach = self.D(fake.detach())
        return d_real, d_fake_detach

    def optimize_parametersD(self, batch_size, data):
        self.set_requires_grad(self.D, True )
        self.set_requires_grad(self.E, False)
        self.set_requires_grad(self.G, False)
        self.optimizer_D.zero_grad()

        z = sample_noise(batch_size, self.zn_dim, self.zc_dim, self.device)[0]
        real = data.to(self.device)
        fake = self.G(z)
        d_real, d_fake_detach = self.forward_D(real, fake)

        if self.loss_metric=='wgan':
            dloss = torch.mean(d_real) - torch.mean(d_fake_detach)
        elif self.loss_metric=='wgan-gp':
            dloss = torch.mean(d_real) - torch.mean(d_fake_detach)\
                 + calc_gradient_penalty(real, fake.detach(), self.D)
        elif self.loss_metric=='vanline':
            label_real = torch.full((fake.size(0), 1), 1, device=self.device)
            label_fake = torch.full((fake.size(0), 1), 0, device=self.device)
            real_loss = self.bceloss(d_real, label_real.float())
            fake_loss = self.bceloss(d_fake_detach, label_fake.float())
            dloss = (real_loss + fake_loss) / 2
        elif self.loss_metric=='hinge':
            dloss = F.relu(1-d_real).mean() + F.relu(1+d_fake_detach).mean()

        dloss.backward()
        self.optimizer_D.step()
        return dloss.item()


    def set_requires_grad(self, nets, requires_grad=False):
        for param in nets.parameters():
            param.requires_grad_(requires_grad)