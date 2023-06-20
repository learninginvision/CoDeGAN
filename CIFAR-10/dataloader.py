import torch
import torchvision
import torchvision.transforms as transforms

def getloader(batch_size, root='./dataset/cifar10', train=True):
    transform = transforms.Compose(
               [transforms.ToTensor(),
                transforms.Lambda(lambda x: x * 2. - 1.),])

    trainset = torchvision.datasets.CIFAR10(root=root, train=train, transform=transform, download=True)
    loader = torch.utils.data.DataLoader\
        (trainset, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=True)
    return loader

    