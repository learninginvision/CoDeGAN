"""
Author: Yonglong Tian (yonglong@mit.edu)
Date: May 07, 2020
"""
from __future__ import print_function

import torch
import torch.nn as nn
from collections import defaultdict

class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all', base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        
    def forward(self, features, mask, logits_mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]


        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        if logits_mask is None:
            logits_mask = torch.scatter(
                torch.ones_like(mask),
                1,
                torch.arange(batch_size * anchor_count).view(-1, 1).to(mask.device),
                0
            )
        else:
            logits_mask = logits_mask.repeat(anchor_count, contrast_count)
            for i in range(logits_mask.shape[0]):
                logits_mask[i][i] = 0
        assert mask.shape == logits_mask.shape
        # mask-out self-contrast cases
        # logits_mask = torch.scatter(
        #     torch.ones_like(mask),
        #     1,
        #     torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
        #     0
        # )
        # print(mask)


        mask = mask * logits_mask
        # print(logits_mask)
        # print(mask)
        # print(weight_mask)
        # exit()
        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        # print(exp_logits)
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        
        # print(log_prob.shape)
        # import numpy as npl
        # result1 = np.array(log_prob.detach().cpu())
        # np.savetxt('npresult1.txt',result1)

        # compute mean of log-likelihood over positive
        # print(mask.sum(1))
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        # mean_log_prob_pos = torch.where(torch.isnan(mean_log_prob_pos),torch.full_like(mean_log_prob_pos,0),mean_log_prob_pos)
        # print(mean_log_prob_pos)
        # exit()

        # print(mean_log_prob_pos.shape)
        # result1 = np.array(mean_log_prob_pos.detach().cpu())
        # np.savetxt('npresult2.txt',result1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        # print(loss)
        loss = loss.view(anchor_count, batch_size).mean()
        
        return loss
