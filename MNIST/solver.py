import torch
import torch.nn as nn
from   torch import autograd
from   utils.util import *
from   utils.sup_contrastive_loss import SupConLoss
from   model import Generator, Discriminator, Encoder


class Solver():
    def __init__(self, config):

        self.zc_dim  = config['zc_dim']
        self.zn_dim  = config['zn_dim']
        self.fc_dim  = config['fc_dim']

        self.beta_1 = config['beta_1']
        self.beta_2 = config['beta_2']

        self.lr      = config['lr']
        self.temp_C  = config['temp_C']
        self.wgan_gp = config['wgan_gp']
        self.device  = 'cuda:{}'.format(config['gpu_id'])

        self.G = Generator(self.zn_dim, self.zc_dim).to(self.device)
        self.D = Discriminator(self.wgan_gp).to(self.device)
        self.E = Encoder(self.zn_dim, self.fc_dim).to(self.device)

        self.optimizer_G = torch.optim.Adam(self.G.parameters(), lr=self.lr, betas=(0.5, 0.9), weight_decay=2.5*1e-5)
        self.optimizer_D = torch.optim.Adam(self.D.parameters(), lr=self.lr, betas=(0.5, 0.9))
        self.optimizer_E = torch.optim.Adam(self.E.parameters(), lr=self.lr, betas=(0.5, 0.9), weight_decay=2.5*1e-5)

        self.mseloss = nn.MSELoss().to(self.device)
        self.celoss  = nn.CrossEntropyLoss().to(self.device)
        self.bceloss = nn.BCELoss().to(self.device)
        self.C_loss  = SupConLoss(temperature=self.temp_C, device=self.device).to(self.device)


    def forward_G(self, z, batch_size):
        fake   = self.G(z)
        d_fake = self.D(fake)
        # add contrastive
        images = torch.cat([fake, train_transform(fake)], dim=0)
        fz, fc = self.E(images)
        f1, f2 = torch.split(fc, [batch_size, batch_size], dim=0)
        fz, _  = torch.split(fz, [batch_size, batch_size], dim=0)
        fc = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)

        if not self.wgan_gp:
            self.lvalid = autograd.Variable(torch.FloatTensor(fake.size(0), 1)\
                .fill_(1.0), requires_grad=False).to(self.device)

        return d_fake, fz, fc
  
    def optimize_parametersG(self, batch_size):
        self.optimizer_G.zero_grad()

        z, zn, zc = sample_noise(batch_size, self.zn_dim, self.zc_dim, self.device)
        d_fake, fz, fc = self.forward_G(z, batch_size)

        ecloss = self.C_loss(fc, zc)
        ezloss = self.mseloss(fz, zn)
        eloss  = self.beta_1*ecloss + self.beta_2*ezloss
        if self.wgan_gp:
            gloss = torch.mean(d_fake)
        else:
            gloss = self.bceloss(d_fake, self.lvalid)

        loss = gloss + eloss
        loss.backward(retain_graph=True)
        self.optimizer_G.step()
        return gloss.item()


    def forward_E(self, z, batch_size):
        # add contrastive
        fake_q = self.G(z)
        fake_k = train_transform(fake_q)

        images =torch.cat([fake_q, fake_k], dim=0)
        fz, fc = self.E(images)
        f1, f2 = torch.split(fc, [batch_size, batch_size], dim=0)
        fz, _  = torch.split(fz, [batch_size, batch_size], dim=0)
        fc = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
        return fz, fc

    def optimize_parametersE(self, batch_size):
        self.optimizer_E.zero_grad()

        z, zn, zc = sample_noise(batch_size, self.zn_dim, self.zc_dim, self.device)
        fz, fc    = self.forward_E(z, batch_size)

        closs = self.C_loss(fc, zc)
        zloss = self.mseloss(fz, zn)
        eloss = closs + (self.beta_2/self.beta_1)*zloss
        
        eloss.backward()
        self.optimizer_E.step()
        return closs.item(), zloss.item()


    def forward_D(self, real, fake):
        d_real = self.D(real)
        d_fake_detach = self.D(fake.detach())

        if not self.wgan_gp:
            self.lvalid = autograd.Variable(torch.FloatTensor(fake.size(0), 1).fill_(1.0), requires_grad=False).to(self.device)
            self.lfake  = autograd.Variable(torch.FloatTensor(fake.size(0), 1).fill_(0.0), requires_grad=False).to(self.device)
        return d_real, d_fake_detach

    def optimize_parametersD(self, batch_size, data):
        self.optimizer_D.zero_grad()

        z = sample_noise(batch_size, self.zn_dim, self.zc_dim, self.device)[0]
        real = data.to(self.device)
        fake = self.G(z)
        d_real, d_fake_detach = self.forward_D(real, fake)

        if self.wgan_gp:
            grad_penalty = calc_gradient_penalty(real, fake, self.D)
            dloss = torch.mean(d_real) - torch.mean(d_fake_detach) + grad_penalty
        else:
            real_loss = self.bceloss(d_real, self.lvalid)
            fake_loss = self.bceloss(d_fake_detach, self.lfake)
            dloss     = (real_loss + fake_loss) / 2

        dloss.backward()
        self.optimizer_D.step()
        return dloss.item()
