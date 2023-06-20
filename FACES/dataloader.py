import torch
import torchvision
import torchvision.transforms as transforms


def getloader(batch_size, root='../faces_tangle', root_real=['../faces_99000/faces_zc_100']):
    transform = transforms.Compose([
                transforms.Grayscale(),
                transforms.Resize(32),
                transforms.ToTensor(),
                ])

    trainset = torchvision.datasets.ImageFolder(root=root, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
    dataset = FacesDataset(root_real, transform=transform)
    return dataset, trainloader


class FacesDataset(torch.utils.data.Dataset):
    def __init__(self, root_real, transform):
        super().__init__()

        self.factor_sizes = [10]
        self.real_dataset = {k: torchvision.datasets.ImageFolder(root=root_real[k], transform=transform) for k in range(len(self.factor_sizes))}
        self.real_data  = {}
        self.real_label = {}
        for k in self.real_dataset.keys():
            _data, _label = [], []
            for data,label in torch.utils.data.DataLoader(self.real_dataset[k]):
                _data.append(data)
                _label.append(label)
            self.real_data[k] = torch.stack(_data)
            self.real_label[k] = torch.stack(_label)

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def sample_fixed_value_batch_from_dataset(self, batch_size, factor_index, factor_value):
        mask = self.real_label[factor_index] == factor_value
        tmp_data = self.real_data[factor_index][mask]
        tmp_labels = self.real_label[factor_index][mask]

        batch_index = list(torch.utils.data.WeightedRandomSampler(weights=[1 for _ in range(tmp_labels.size(0))], num_samples=batch_size, replacement=False))
        batch_data = tmp_data[batch_index].float()
        batch_labels = tmp_labels[batch_index]
        # Expand channel dim
        if batch_data.dim() == 3:
            batch_data = batch_data.unsqueeze(1)
        return batch_data, batch_labels

    def sample_paired_all_value_batch(self, batch_size, factor_index):
        data, labels = [], []
        for factor_value in range(self.factor_sizes[factor_index]):
            _data, _label = self.sample_fixed_value_batch_from_dataset(batch_size, factor_index, factor_value)
            data.append(_data[:batch_size])
            labels.append(_label[:batch_size])
        data = torch.cat(data, dim=0)
        labels = torch.cat(labels, dim=0)
        return [data, labels]
