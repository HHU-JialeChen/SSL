# -*- coding: utf-8 -*-
# @Time    : 2021-12-11 16:51
# @Author  : Wily
# @File    : losses.py
# @Software: PyCharm


from __future__ import print_function

import torch
import torch.nn as nn


"""Reference: Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf."""
class SupCluLoss(nn.Module):
    def __init__(self, temperature=0.3, contrast_mode='all',
                 base_temperature=0.07):
        super(SupCluLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None):
        device = torch.device('cuda')
                  

        batch_size = features.shape[0]
        
        labels = labels.contiguous().view(-1, 1)
        #print('label_shape',labels.shape)
        #print('batch_size',batch_size)
        if labels.shape[0] != batch_size:
            #print(labels.shape[0],batch_size)
            raise ValueError('Num of labels does not match num of features')
        mask = torch.eq(labels, labels.T).float().to(device)


        contrast_count = 1
        contrast_feature = features

        anchor_feature = contrast_feature
        anchor_count = contrast_count

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask
        
        #cosine
        #a = features / torch.norm(features, dim=-1, keepdim=True)  
        #similarity = torch.mm(a, a.T)
        #similarity = similarity * mask
        #similarity_mask = torch.where(similarity < 0.3, 0.0, 1.0)

        # compute log_prob

        exp_logits = torch.exp(logits) * logits_mask
        #print("exp_logits",exp_logits.shape)
        postive_logits = torch.exp(logits) * mask
        #print("postive_logits",postive_logits.shape)
        log_prob = torch.log(postive_logits.sum(1, keepdim=True)) - torch.log(exp_logits.sum(1, keepdim=True))
        #print("log_prob",log_prob.shape)
        mean_log_prob_pos = log_prob.sum(1) / mask.sum(1)
        #print("mean_log_prob_pos",mean_log_prob_pos.shape)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        #loss = - log_prob
        loss = loss.view(anchor_count, batch_size).mean()

        return loss



