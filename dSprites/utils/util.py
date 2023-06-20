import torch
import torchvision.transforms as transforms
from   torch import autograd


# sample noises
def sample_noise(batch_size, zn_dim, zc_dim, device='cpu'):
    zn = torch.randn(batch_size, zn_dim)

    m_zeros = torch.zeros(batch_size, zc_dim)
    zc_idx  = torch.tensor(list(torch.utils.data.WeightedRandomSampler\
        (weights=[1 for _ in range(zc_dim)], num_samples=batch_size, replacement=True)))
    zc = m_zeros.scatter_(1,zc_idx.view(-1,1),1)

    z = torch.cat((zn, zc), dim=1)
    if device != 'cpu':
        return z.to(device), zn.to(device), zc_idx.to(device)

    return z, zn, zc_idx


# generate mask for contrastive loss
def get_mask_zc_discrete(batch_size, labels=None):
    if labels is None:
        mask = torch.eye(batch_size, dtype=torch.float32)
    else:
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float()
    return mask.float()


# calculate gradient penalty
def calc_gradient_penalty(real_data, generated_data, D):
    LAMBDA = 10
    device = real_data.device
    b_size = real_data.size()[0]
    alpha  = torch.rand(b_size, 1, 1, 1)
    alpha  = alpha.expand_as(real_data)
    alpha  = alpha.to(device)

    interpolated = alpha * real_data.data + (1 - alpha) * generated_data.data
    interpolated = autograd.Variable(interpolated, requires_grad=True)
    interpolated = interpolated.to(device)

    prob_interpolated = D(interpolated)
    gradients = autograd.grad(outputs=prob_interpolated, inputs=interpolated,
                        grad_outputs=torch.ones(prob_interpolated.size()).to(device),
                        create_graph=True, retain_graph=True)[0]

    gradients = gradients.view(b_size, -1)
    gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)
    return LAMBDA * ((gradients_norm - 1) ** 2).mean()


# image augmentation
def train_transform(image):
    transform = transforms.Compose([
        transforms.RandomResizedCrop(size=64, scale=(0.8, 1.)),
    ])

    return transform(image).detach()


# change tensor to array
def to_numpy(tensor):
    if isinstance(tensor, (int, float)):
        return tensor
    else:
        return tensor.data.cpu().numpy()
