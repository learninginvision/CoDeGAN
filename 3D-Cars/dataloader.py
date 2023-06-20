import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from   torch.utils.data import Dataset, DataLoader


class Cars3D(Dataset):
    def __init__(self, root, train_class):
        datasets = np.load(root + "cars3d-x64_apart_by_class.npz")
        imgs = datasets['imgs']
        factors = datasets['factors']
        num_class = len(imgs)
        
        index = [54, 141, 4, 7, 1, 25, 26, 24, 43, 23]
        self.train_imgs = []
        self.train_factors = []
        for i in range(len(index)):
            for j in range(len(imgs[index[i]])):
                self.train_imgs.append(np.squeeze(imgs[index[i]][j]))
                self.train_factors.append(i)
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


def getloader(batch_size, root='./dataset/cars3d/', train_class=10):
    trainset = Cars3D(root=root, train_class=train_class)
    loader = DataLoader\
        (trainset, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=True)
    print('successfully loaded {} cars3d data'.format(len(trainset)))
    return loader
