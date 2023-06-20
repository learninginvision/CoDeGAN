import os
import torch
import torchvision
import numpy as np
import torchvision.transforms as transforms


def getloader(batch_size, root='../1107_dsprites', root_real=['../1107_dsprites/1107_dsprites_real500']):
    trainset = DSpritesDataset(root, root_real)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True)
    return trainset, trainloader


def get_testloader(batch_size, root='../1107_dsprites/1107_dsprites_test'):
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor(),
    ])

    trainset = torchvision.datasets.ImageFolder(root, transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size)
    return trainloader


class DSpritesDataset(torch.utils.data.Dataset):
    def __init__(self, root, root_real):
        super().__init__()

        filename = "dsprites_ndarray_train.npz"
        path = os.path.join(root, filename)
        with np.load(path, encoding="latin1", allow_pickle=True) as dataset:
            data = torch.tensor(dataset["imgs"])
            targets = torch.tensor(dataset["latents_classes"])

        self.data = data
        self.targets = targets
        self.factor_sizes = [3]

        self.real_dataset = {k: torchvision.datasets.ImageFolder(root=root_real[k], transform=transforms.Compose([transforms.Grayscale(), transforms.ToTensor()])) for k in range(len(self.factor_sizes))}
        self.real_data = {}
        self.real_label = {}
        for k in self.real_dataset.keys():
            _data, _label = [], []
            for data, label in torch.utils.data.DataLoader(self.real_dataset[k]):
                _data.append(data)
                _label.append(label)
            self.real_data[k] = torch.stack(_data)
            self.real_label[k] = torch.stack(_label)

    def __getitem__(self, index):
        return self.data[index].unsqueeze(0).float(), self.targets[index]

    def __len__(self):
        return self.data.size(0)

    def sample_fixed_value_batch_from_dataset(self, batch_size, factor_index, factor_value):
        mask = self.real_label[factor_index] == factor_value
        tmp_data = self.real_data[factor_index][mask]
        tmp_targets = self.real_label[factor_index][mask]

        batch_index = list(torch.utils.data.WeightedRandomSampler(weights=[1 for _ in range(tmp_targets.size(0))], num_samples=batch_size, replacement=False))
        batch_data = tmp_data[batch_index].float()
        batch_targets = tmp_targets[batch_index]
        # Expand channel dim
        if batch_data.dim() == 3:
            batch_data = batch_data.unsqueeze(1)
        return batch_data, batch_targets

    def sample_paired_all_value_batch(self, batch_size, factor_index):
        data, targets = [], []
        for factor_value in range(self.factor_sizes[factor_index]):
            _data, _label = self.sample_fixed_value_batch_from_dataset(batch_size, factor_index, factor_value)
            data.append(_data)
            targets.append(_label)
        data = torch.cat(data, dim=0)
        targets = torch.cat(targets, dim=0)
        return data, targets

