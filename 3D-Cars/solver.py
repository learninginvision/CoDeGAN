import torch
import torch.nn as nn
import torch.nn.functional as F
from   utils.util import *
from   utils.sup_contrastive_loss import SupConLoss
from   models.Generator import Generator
from   models.Discriminator import Discriminator
from   models.Encoder import EncoderC, EncoderZ, Classifier


class Solver():
    def __init__(self, config):

        self.zn_dim = config['zn_dim']
        self.zc_dim = config['zc_dim']
        self.fc_dim = config['fc_dim']

        self.beta_2 = config['beta_2']
        self.beta_1 = config['beta_1']

        self.lr          = config['lr']
        self.temp_C      = config['temp_C']
        self.loss_metric = config['loss_metric']
        self.device = 'cuda:{}'.format(config['gpu_id'])

        self.D  = Discriminator().to(self.device)
        self.G  = Generator(dim_z=self.zn_dim + self.zc_dim).to(self.device)
        self.EC = EncoderC(name='resnet18', feat_dim=self.fc_dim).to(self.device)
        self.EZ = EncoderZ(name='resnet18', feat_dim=self.zn_dim).to(self.device)
        self.Classifier = Classifier().to(self.device)

        self.optimizer_G  = torch.optim.Adam(self.G.parameters(),  lr=self.lr, betas=(0.5, 0.999))
        self.optimizer_D  = torch.optim.Adam(self.D.parameters(),  lr=self.lr, betas=(0.5, 0.999))
        self.optimizer_EC = torch.optim.Adam(self.EC.parameters(), lr=self.lr, betas=(0.5, 0.999))
        self.optimizer_EZ = torch.optim.Adam(self.EZ.parameters(), lr=self.lr, betas=(0.5, 0.999))

        self.bceloss = nn.BCELoss().to(self.device)
        self.mseloss = nn.MSELoss().to(self.device)
        self.C_loss  = SupConLoss(device=self.device, temperature=self.temp_C).to(self.device)


    def forward_G(self, z, zc, batch_size):
        img_q  = self.G(z)
        d_fake = self.D(img_q)

        # add contrastive
        img_k = train_transform(img_q).to(self.device)
        fz = self.EZ(img_q)

        # # add few_labels
        if real_data is not None:
            (real_q, real_label) = real_data
            real_q = real_q.to(self.device)
            real_k = train_transform(real_q, self.trans_onpil).to(self.device)
            zc     = torch.cat([zc, real_label.to(self.device)], dim=0)
            img_q  = torch.cat([img_q, real_q], dim=0)
            img_k  = torch.cat([img_k, real_k], dim=0)
            batch_size += real_q.size(0)

        images = torch.cat([img_q, img_k], dim=0)
        fc = self.EC(images)
        f1, f2 = torch.split(fc, [batch_size, batch_size], dim=0)
        fc = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)

        return d_fake, fz, fc, zc

    def optimize_parametersG(self, batch_size):
        self.set_requires_grad(self.D,  False)
        self.set_requires_grad(self.G,  True )
        self.set_requires_grad(self.EC, False)
        self.set_requires_grad(self.EZ, False)
        self.optimizer_G.zero_grad()

        z, zn, zc = sample_noise(batch_size, self.zn_dim, self.zc_dim, self.device)
        d_fake, fz, fc, zc = self.forward_G(z, zc, batch_size)

        ecloss = self.C_loss( fc, zc)
        ezloss = self.mseloss(fz, zn)
        eloss  = self.beta_1 * ecloss + self.beta_2 * ezloss

        if self.loss_metric == 'hinge':
            gloss = - torch.mean(d_fake.view(-1))
        elif self.loss_metric == 'wgan-gp':
            gloss = torch.mean(d_fake)
        else:
            raise ValueError('Unknown Loss Metric')
        
        loss = gloss + eloss
        loss.backward()
        self.optimizer_G.step()
        return gloss.item()


    def forward_EC(self, z, zc, batch_size):
        # add contrastive
        img_q = self.G(z)
        img_k = train_transform(img_q).to(self.device)

        images = torch.cat([img_q, img_k], dim=0)
        fc = self.EC(images)
        f1, f2 = torch.split(fc, [batch_size, batch_size], dim=0)
        fc = torch.cat([f1.unsqueeze(1),f2.unsqueeze(1)],dim=1)
        return fc, zc

    def optimize_parametersEC(self, batch_size):
        self.set_requires_grad(self.D,  False)
        self.set_requires_grad(self.G,  False)
        self.set_requires_grad(self.EC, True )
        self.set_requires_grad(self.EZ, False)
        self.optimizer_EC.zero_grad()

        z, _, zc = sample_noise(batch_size, self.zn_dim, self.zc_dim, self.device)
        fc, zc   = self.forward_EC(z, zc, batch_size)

        closs = self.C_loss(fc, zc)

        closs.backward()
        self.optimizer_EC.step()
        return closs.item()


    def forward_EZ(self, z):
        fake = self.G(z)
        fz   = self.EZ(fake)
        return fz

    def optimize_parametersEZ(self, batch_size):
        self.set_requires_grad(self.D,  False)
        self.set_requires_grad(self.G,  False)
        self.set_requires_grad(self.EC, False)
        self.set_requires_grad(self.EZ, True )
        self.optimizer_EZ.zero_grad()

        z, zn, _ = sample_noise(batch_size, self.zn_dim, self.zc_dim, self.device)
        fz = self.forward_EZ(z)

        zloss = self.mseloss(fz, zn)

        zloss.backward()
        self.optimizer_EZ.step()
        return zloss.item()


    def forward_D(self, real, fake):
        d_real        = self.D(real)
        d_fake_detach = self.D(fake.detach())
        return d_real, d_fake_detach

    def optimize_parametersD(self, batch_size, data):
        self.set_requires_grad(self.D,  True )
        self.set_requires_grad(self.G,  False)
        self.set_requires_grad(self.EC, False)
        self.set_requires_grad(self.EZ, False)
        self.optimizer_D.zero_grad()

        z = sample_noise(batch_size, self.zn_dim, self.zc_dim, self.device)[0]
        real = data.to(self.device)
        fake = self.G(z)
        d_real, d_fake_detach = self.forward_D(real, fake)

        if   self.loss_metric == 'hinge':
            dloss = torch.mean(F.relu(1-d_real.view(-1))) + torch.mean(F.relu(1+d_fake_detach.view(-1)))
        elif self.loss_metric == 'wgan-gp':
            dloss = torch.mean(d_real) - torch.mean(d_fake_detach) + calc_gradient_penalty(real, fake.detach(), self.D)
        else:
            raise ValueError('Unknown Loss Metric')

        dloss.backward()
        self.optimizer_D.step()
        return dloss.item()


    def set_requires_grad(self, nets, requires_grad=False):
        for param in nets.parameters():
            param.requires_grad_(requires_grad)
