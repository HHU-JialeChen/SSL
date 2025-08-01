#!/usr/bin/env python
import pickle as pkl
import sys

import torch


if __name__ == "__main__":
    input = "pretrained weights path"

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

    res = {"model": newmodel,  "matching_heuristics": True}

    torch.save(res,"new_weight.pth")
