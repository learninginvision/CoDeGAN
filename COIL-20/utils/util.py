import torch
import torchvision.transforms as transforms
import torch.distributions.categorical as categorical
from   torch import autograd


# sample noises
def sample_noise(batch_size, zn_dim, zc_dim, device='cpu'):
    set_index = categorical.Categorical(torch.tensor([1/zc_dim]*zc_dim))

    zn      = torch.randn(batch_size, zn_dim)
    m_zeros = torch.zeros(batch_size, zc_dim)
    zc_idx  = [set_index.sample().view(-1,1) for _ in range(batch_size)]
    zc_idx  = torch.cat(zc_idx, dim=0)
    zc      = m_zeros.scatter_(1, zc_idx, 1)
    z       = torch.cat((zn, zc), dim=1)
    if device != 'cpu':
        return z.to(device), zn.to(device), zc_idx.squeeze().to(device)

    return z, zn, zc_idx.squeeze()


# sample noise with fixed zc
def sample_noise_with_label(batch_size, zn_dim, zc_dim, device, labels):
    zn = torch.randn(batch_size, zn_dim).to(device)
    zc = labels * torch.ones((batch_size, 1)).long().to(device)
    zc_dix = zc.float()
    m_zeros = torch.zeros(batch_size, zc_dim).to(device)
    zc = m_zeros.scatter_(1, zc, 1)
    z = torch.cat((zn, zc), dim=1).to(device)
    return z, zc_dix


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
def train_transform(image, onpil=False):
    transform = transforms.Compose([
        transforms.Lambda(lambda x: (x + 1.)/ 2.),
        transforms.RandomHorizontalFlip(),
        transforms.Lambda(lambda x: x * 2. - 1.),
    ])

    image_tran = transform(image)

    return image_tran.detach()

