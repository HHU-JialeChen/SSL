# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import diffdist
import math
import torch.nn as nn
from .new3_losses import SupCluLoss
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output

class SimSiam(nn.Module):
    """
    Build a SimSiam model.
    """
    def __init__(self, base_encoder, dim=2048, pred_dim=512):
        """
        dim: feature dimension (default: 2048)
        pred_dim: hidden dimension of the predictor (default: 512)
        """
        super(SimSiam, self).__init__()

        # create the encoder
        # num_classes is the output fc dimension, zero-initialize last BNs
        self.encoder = base_encoder(num_classes=dim, zero_init_residual=True)

        # build a 3-layer projector
        prev_dim = self.encoder.fc.weight.shape[1]
        self.encoder.fc = nn.Sequential(nn.Linear(prev_dim, prev_dim, bias=False),
                                        nn.BatchNorm1d(prev_dim),
                                        nn.ReLU(inplace=True), # first layer
                                        nn.Linear(prev_dim, prev_dim, bias=False),
                                        nn.BatchNorm1d(prev_dim),
                                        nn.ReLU(inplace=True), # second layer
                                        self.encoder.fc,
                                        nn.BatchNorm1d(dim, affine=False)) # output layer
        self.encoder.fc[6].bias.requires_grad = False # hack: not use bias as it is followed by BN

        # build a 2-layer predictor
        self.predictor = nn.Sequential(nn.Linear(dim, pred_dim, bias=False),
                                        nn.BatchNorm1d(pred_dim),
                                        nn.ReLU(inplace=True), # hidden layer
                                        nn.Linear(pred_dim, dim)) # output layer
        self.criterion = nn.CosineSimilarity(dim=1)
        self.criterion_clu = SupCluLoss(temperature=0.3)
    @torch.no_grad()
    def _batch_gather_ddp(self, images):
        """
        gather images from different gpus and shuffle between them
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        images_gather = []
        for i in range(4):
            #print(i, np.array(images).shape)
            #print(images[i].shape)#128*3*112*112
            batch_size_this = images[i].shape[0]
            images_gather.append(concat_all_gather(images[i]))
            batch_size_all = images_gather[i].shape[0]
        num_gpus = batch_size_all // batch_size_this

        n, c, h, w = images_gather[0].shape                          #images_gather[0]=128*3*224*224
        permute = torch.randperm(n * 4).cuda()          #512

        torch.distributed.broadcast(permute, src=0)
        images_gather = torch.cat(images_gather, dim=0)                #512*3*224*224
        images_gather = images_gather[permute, :, :, :]
        #col1 = torch.cat([images_gather[0:n], images_gather[n:2 * n]], dim=3)
        #col2 = torch.cat([images_gather[2 * n:3 * n], images_gather[3 * n:4 * n]], dim=3)
        #images_gather = torch.cat([col1, col2], dim=2)                #256*3*224*224

        bs = images_gather.shape[0] // num_gpus
        gpu_idx = torch.distributed.get_rank()

        return images_gather[bs * gpu_idx:bs * (gpu_idx + 1)], permute,permute[bs * gpu_idx:bs * (gpu_idx + 1)], n #256->128 

    @torch.no_grad()
    def _batch_gather_ddp2(self, images, permute):
        """
        gather images from different gpus and shuffle between them
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        images_gather = []
        for i in range(4):
            #print(i)
            batch_size_this = images[i].shape[0]
            images_gather.append(concat_all_gather(images[i]))
            batch_size_all = images_gather[i].shape[0]
        num_gpus = batch_size_all // batch_size_this

        n, c, h, w = images_gather[0].shape
        torch.distributed.broadcast(permute, src=0)
        images_gather = torch.cat(images_gather, dim=0)
        images_gather = images_gather[permute, :, :, :]
        #col1 = torch.cat([images_gather[0:n], images_gather[n:2 * n]], dim=3)
        #col2 = torch.cat([images_gather[2 * n:3 * n], images_gather[3 * n:4 * n]], dim=3)
        #images_gather = torch.cat([col1, col2], dim=2)

        bs = images_gather.shape[0] // num_gpus
        gpu_idx = torch.distributed.get_rank()

        return images_gather[bs * gpu_idx:bs * (gpu_idx + 1)], permute, n
        
    def forward_(self, q):#128*2048*1*1
        q_gather = concat_all_gather_crop(q)
        n,c = q_gather.shape
        q_gather = q_gather.view(n,-1)

        return q_gather    
    def forward(self, x1, x2):
        """
        Input:
            x1: first views of images
            x2: second views of images
        Output:
            p1, p2, z1, z2: predictors and targets of the network
            See Sec. 3 of https://arxiv.org/abs/2011.10566 for detailed notations
        """
        images_gather1, permute1,p_rank, bs_all1 = self._batch_gather_ddp(x1)
        images_gather2, permute2, bs_all2 = self._batch_gather_ddp2(x2, permute1)

        # compute features for one view
        z1 = self.encoder(images_gather1) # NxC
        z2 = self.encoder(images_gather2) # NxC

        p1 = self.predictor(z1) # NxC
        p2 = self.predictor(z2) # NxC
        
        con_loss = -(self.criterion(p1, z2).mean() + self.criterion(p2, z1).mean()) * 0.5
        
        #clu
        label_crop_ = p_rank % bs_all1
        #print(p_rank)
        #print(bs_all1)
        #print(label_crop_)
        #p1= self.forward_(p1)
        #p2= self.forward_(p2)
        #z1= self.forward_(z1)
        #z2= self.forward_(z2)
        label_crop = torch.cat([label_crop_,label_crop_], dim=0)
        logits_1 = torch.cat([p1, z2], dim=0)
        logits_1 = nn.functional.normalize(logits_1, dim=1)
        logits_2 = torch.cat([p2, z1], dim=0)
        logits_2 = nn.functional.normalize(logits_2, dim=1)
        clu_loss = (self.criterion_clu(logits_1, label_crop)+self.criterion_clu(logits_2, label_crop))/2
        
        return con_loss,clu_loss#p1, p2, z1.detach(), z2.detach()
def concat_all_gather_crop(tensor):
    tensors_gather = [torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())]
    tensors_gather = diffdist.functional.all_gather(tensors_gather, tensor, next_backprop=None, inplace=True)
    output = torch.cat(tensors_gather, dim=0)
    return output