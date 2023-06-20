import torch
import torchvision
import matplotlib.pyplot as plt
from   sklearn import manifold


# visualize generated images
def save_fake_image(batch_size, G, zn_dim, zc_dim, device):
    zn =  torch.randn(batch_size, zn_dim).to(device)

    fake_image_zc1 = []
    for labels in range(zc_dim):
        m_zeros = torch.zeros(batch_size, zc_dim)
        zc_idx = labels * torch.ones((batch_size, 1)).long()
        zc = m_zeros.scatter_(1, zc_idx, 1)
        fake_image_zc1.append(G(torch.cat((zn, zc.to(device)), dim=1).to(device)))

    return torch.cat(fake_image_zc1,dim=0)

def visualize_result(epoch, model_dir, G, zn_dim, zc_dim, device):
    stack_imgs = save_fake_image(10, G, zn_dim, zc_dim, device)
    torchvision.utils.save_image(stack_imgs, \
         model_dir + '/generated_dsprites_epoch{}.png'.format(epoch), nrow=10, normalize=True)


# draw t-SNE image for test if needed
def draw_tsne(X, y):
    '''t-SNE'''
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)
    X_tsne = tsne.fit_transform(X.data.cpu())

    print("Org data dimension is {}. Embedded data dimension is {}".format(X.shape[-1], X_tsne.shape[-1]))

    # visualize t-SNE
    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = (X_tsne - x_min) / (x_max - x_min)
    for i in range(10):
        plt.scatter(X_norm[y==i,0], X_norm[y==i,1], alpha=0.8, label='%s' % i)
    ax = plt.gca()
    ax.spines['bottom'].set_linewidth(3)
    ax.spines['top'].set_linewidth(3)
    ax.spines['left'].set_linewidth(3)
    ax.spines['right'].set_linewidth(3)

    plt.tick_params(labelsize=16)
    plt.legend()
    plt.legend(loc='right', bbox_to_anchor=(1.15,0.5), prop = {'weight' : 'normal', \
        'size' : 16,}, handlelength = 0.4)
    plt.savefig('./t-SNE_result.png')
    plt.show()
