#!/usr/bin/env python
import pickle as pkl
import sys

import torch


if __name__ == "__main__":
    input = "/home/user/cjl/Aug_simsiam_clu/aug_loss3_e400_cross03_coco_b_124_lr_0.05/checkpoint_40.pth.tar"

    obj = torch.load(input, map_location="cpu")
    obj = obj["state_dict"]

    newmodel = {}
    for k, v in obj.items():
        old_k = k
        if "layer" not in k:
            k = "backbone.stem." + k.replace("module.encoder.","")
        for t in [1, 2, 3, 4]:
            k = k.replace("module.encoder.layer{}".format(t), "backbone.res{}".format(t + 1))
        for t in [1, 2, 3]:
            k = k.replace("bn{}".format(t), "conv{}.norm".format(t))
        k = k.replace("downsample.0", "shortcut")
        k = k.replace("downsample.1", "shortcut.norm")
        print(old_k, "->", k)
        newmodel[k] = v.numpy()

    res = {"model": newmodel, "__author__": "MOCO", "matching_heuristics": True}

    torch.save(res,"/home/user/cjl/Aug_simsiam_clu/aug_loss3_e400_cross03_coco_b_124_lr_0.05/loss3_04_b_124_e400.pth.tar")
