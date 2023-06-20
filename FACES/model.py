import torch.nn as nn
import torch.nn.functional as F


# Weight Initializer
def initialize_weights(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.ConvTranspose2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()

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
    def __init__(self, zn_dim, zc_dim):
        super(Generator, self).__init__()
        self.zn_dim = zn_dim
        self.zc_dim = zc_dim
        self.fc = nn.Sequential(
            nn.Linear(self.zc_dim+self.zn_dim,1024),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(1024),
            nn.Linear(1024,128*8*8),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(128*8*8),
        )
        self.cnn = nn.Sequential(
            nn.ConvTranspose2d(128,64,kernel_size=4,stride=2,padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64,1,kernel_size=4,stride=2,padding=1),
            nn.Sigmoid(),
        )
        initialize_weights(self)

    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, 128, 8, 8)
        x = self.cnn(x)
        return x

    
class Encoder(nn.Module):
    def __init__(self, fz_dim, fc_dim):
        super(Encoder, self).__init__()
        self.fz_dim = fz_dim
        self.fc_dim = fc_dim
        self.cnn = nn.Sequential(
            nn.Conv2d(1,64,kernel_size=4,stride=2,padding=1),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(64,128,kernel_size=4,stride=2,padding=1),
            # nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2,inplace=True),
        )
        self.fc = nn.Sequential(
            nn.Linear(128*8*8,1024),
            # nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Linear(1024,self.fz_dim+self.fc_dim),
        )
        initialize_weights(self)
        
    def forward(self, x):
        x = self.cnn(x)
        x = x.view(-1, 128*8*8)
        x = self.fc(x)
        zn = x[:, :self.fz_dim]
        zc = x[:, self.fz_dim:]
        return zn, F.normalize(zc, dim=1)
        
class Discriminator(nn.Module):
    def __init__(self, wass_metric='vanline'):
        super(Discriminator, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1,64,kernel_size=4,stride=2,padding=1),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(64,128,kernel_size=4,stride=2,padding=1),
            # nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2,inplace=True),
        )
        self.fc = nn.Sequential(
            nn.Linear(128*8*8,1024),
            # nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Linear(1024,1),
        )
        if wass_metric=='vanline':
            self.fc = nn.Sequential(self.fc,nn.Sigmoid())
        initialize_weights(self)

    def forward(self, x):
        x = self.cnn(x)
        x = x.view(-1, 128*8*8)
        x = self.fc(x)
        return x