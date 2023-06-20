import torch
import torchvision
import torchvision.transforms as transforms

def getloader(batch_size, root='./dataset/mnist', train=True):
    transform = transforms.Compose([
                transforms.ToTensor(),
                ])

    trainset = torchvision.datasets.MNIST(root=root, train=train, transform=transform, download=True)
    trainloader = torch.utils.data.DataLoader\
        (trainset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
    return trainloader
