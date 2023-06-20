import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from   torch.utils.data import Dataset, DataLoader


class COIL20(Dataset):
    def __init__(self, root, train_class):
        datasets = np.load(root + "coil20.npz")
        imgs = datasets['images']
        factors = datasets['labels']
        num_class = train_class
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


def getloader(batch_size, root='./dataset/coil20/', train_class=20):
    trainset = COIL20(root=root, train_class=train_class)
    loader = DataLoader\
        (trainset, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=True)
    print('successfully loaded {} coil-20 data'.format(len(trainset)))
    return loader