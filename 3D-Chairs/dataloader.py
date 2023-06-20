import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from   torch.utils.data import Dataset, DataLoader


class Chairs3D(Dataset):
    def __init__(self, root, train_class):
        datasets = np.load(root + "chairs3d_v1.npz")
        imgs = datasets['images']
        factors = datasets['labels']
        num_class = 10
        self.train_imgs = []
        self.train_factors = []
        for i in range(len(imgs)):
            self.train_imgs.append(np.squeeze(imgs[i]))
            self.train_factors.append(factors[i])
        del imgs
        del factors
        
        self.tranforms =  transforms.Compose(
               [
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x * 2. - 1.),
                ])
        
    def __getitem__(self, index):
        return self.tranforms(self.train_imgs[index]), self.train_factors[index]
    
    def __len__(self):
        return len(self.train_imgs)


def getloader(batch_size, root='./dataset/chairs3d/', train_class=10):
    trainset = Chairs3D(root=root, train_class=train_class)
    loader = DataLoader\
        (trainset, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=True)
    print('successfully loaded {} chairs3d data'.format(len(trainset)))
    return loader
