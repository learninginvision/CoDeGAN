import torch.nn as nn
import torch.nn.functional as F
import math

def pytorch_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.kaiming_uniform_(m.weight.data, a=math.sqrt(5), mode='fan_in', nonlinearity='leaky_relu')
        nn.init.constant_(m.bias.data, 0)
    elif classname.find('ConvTranspose') != -1:
        nn.init.kaiming_uniform_(m.weight.data, a=math.sqrt(5), mode='fan_in', nonlinearity='leaky_relu')
        nn.init.constant_(m.bias.data, 0)
    elif classname.find('Linear') != -1:
        nn.init.kaiming_uniform_(m.weight.data, a=math.sqrt(5), mode='fan_in', nonlinearity='leaky_relu')
        nn.init.constant_(m.bias.data, 0)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def keras_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.xavier_uniform_(m.weight.data, gain=1)
        nn.init.constant_(m.bias.data, 0)
    elif classname.find('ConvTranspose') != -1:
        nn.init.xavier_uniform_(m.weight.data, gain=1)
        nn.init.constant_(m.bias.data, 0)
    elif classname.find('Linear') != -1:
        nn.init.xavier_uniform_(m.weight.data, gain=1)
        nn.init.constant_(m.bias.data, 0)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    elif classname.find('ConvTranspose') != -1:
        nn.init.normal_(m.weight.data, 0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    elif classname.find('Linear') != -1:
        nn.init.normal_(m.weight.data, 0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class Generator(nn.Module):
    def __init__(self, z_dim, fc_dim, im_size=64, ch=1, _init='weight'):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(fc_dim+z_dim, 128),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(128),
            nn.Linear(128,64*4*4),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(64*4*4),
        )
        self.cnn = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32, ch, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )
        if _init=='pytorch':
            self.apply(pytorch_init)
        elif _init=='keras':
            self.apply(keras_init)
        else:
            self.apply(weights_init)

    def forward(self, z):
        h = self.fc(z)
        h = h.view(h.shape[0], 64, 4, 4)
        return self.cnn(h)


class Discriminator(nn.Module):
    def __init__(self, im_size=64, ch=1, loss_metric='vanline', _init='pytorch'):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(ch, 32, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.1, inplace=True),
            nn.utils.spectral_norm(nn.Conv2d(32, 32, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.1, inplace=True),
            nn.utils.spectral_norm(nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.1, inplace=True),
            nn.utils.spectral_norm(nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.fc = nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(64*4*4, 128)),
            nn.LeakyReLU(0.1, inplace=True),
            nn.utils.spectral_norm(nn.Linear(128, 1)),
        )
        if loss_metric=='vanline':
            self.fc = nn.Sequential(
                self.fc,
                nn.Sigmoid(),
            )

        if _init=='pytorch':
            self.apply(pytorch_init)
        elif _init=='keras':
            self.apply(keras_init)
        else:
            self.apply(weights_init)

    def forward(self, x):
        h = self.cnn(x)
        h = h.view(h.shape[0], -1)
        return self.fc(h)


class Encoder(nn.Module):
    def __init__(self, fz_dim, fc_dim, im_size=64, ch=1, _init='pytorch'):
        super().__init__()
        self.fz_dim = fz_dim
        self.cnn1 = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(ch, 32, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.1, inplace=True),
            nn.utils.spectral_norm(nn.Conv2d(32, 32, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.1, inplace=True),
            nn.utils.spectral_norm(nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.1, inplace=True),
            nn.utils.spectral_norm(nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.fc1 = nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(64*4*4, 128)),
            nn.LeakyReLU(0.1, inplace=True),
            nn.utils.spectral_norm(nn.Linear(128, 128)),
            nn.LeakyReLU(0.1, inplace=True),
            nn.utils.spectral_norm(nn.Linear(128, fz_dim+fc_dim))
        )

        if _init=='pytorch':
            self.apply(pytorch_init)
        elif _init=='keras':
            self.apply(keras_init)
        else:
            self.apply(weights_init)

    def forward(self, x):
        h1 = self.cnn1(x)
        h1 = h1.view(h1.shape[0], -1)
        h1 = self.fc1(h1)
        zn = h1[:, :self.fz_dim]
        zc = h1[:, self.fz_dim:]
        return zn, F.normalize(zc, dim=1)