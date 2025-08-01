# Self-Supervised Contrastive Learning for Multi-Label Images

### Introduction
Code for “Self-Supervised Contrastive Learning for Multi-Label Images”. This paper was submitted to CVPR 2023 but unfortunately was not accepted. While we acknowledge that it is not an outstanding or highly robust work and may even seem somewhat outdated, we still believe that the initial idea and motivation are valid and accurate. Moreover, the method is simple and easy to implement. In particular, we are convinced that there is significant potential for exploration in leveraging large-scale multi-label images. We welcome any criticism, suggestions, and guidance from the community.

The code provided here is not the final polished version, but it does include all the major data augmentation operations and the relevant IALoss components for processing COCO images. We hope this will meet the needs of those who wish to build upon our work.

### Preparation

Install PyTorch and download the COCO dataset. Similar to [SimSiam](https://github.com/facebookresearch/simsiam.git), the code release contains minimal modifications for both unsupervised pre-training and linear classification to that code.

### Unsupervised Pre-Training

To do unsupervised pre-training of a ResNet-50 model on COCO in an 2-gpu machine, run:

```
python main/coco_main_simsiam.py
```

### Linear Classification

With a pre-trained model, to train a supervised linear classifier on frozen features/weights, run:

```
python main_lincls.py 
```

### Transfer Validation

We provide here the profiles used for transfer validation of COCO and VOC datasets. Meanwhile, we list the public pre-training weights used, as shown in the table below. Besides, if you need more code samples about our research, we would be delighted to provide them. 

| Method                                                 | Dataset  | Epoch | Batch Size | Download Link                                                                                                 |
| ------------------------------------------------------ | -------- | ----- | ---------- | ------------------------------------------------------------------------------------------------------------- |
| [SimSiam](https://github.com/facebookresearch/simsiam) | ImageNet | 100   | 512        | [Download](https://dl.fbaipublicfiles.com/simsiam/models/100ep/pretrain/checkpoint_0099.pth.tar)              |
| [MoCoV2](https://github.com/facebookresearch/moco)     | ImageNet | 200   | 256        | [Download](https://dl.fbaipublicfiles.com/moco/moco_checkpoints/moco_v2_200ep/moco_v2_200ep_pretrain.pth.tar) |
| [DenseCL](https://github.com/WXinlong/DenseCL)         | COCO     | 800   | 256        | [Download](https://cloudstor.aarnet.edu.au/plus/s/W5oDyYB218xz625/download)                                   |

Transfer validation configurations on the COCO as well as VOC datasets can be found in the ```.\configs\COCO_1x\coco_R_50_C4_1x_sia.yaml``` and ```.\configs\VOC_24k\voc_R_50_C4_24k_sia.yaml```. The same  transfer validation  as for [MoCo](https://github.com/facebookresearch/moco) is performed, more details please see [moco/detection](https://github.com/facebookresearch/moco/tree/master/detection).

To do the weight conversion, run:

```
python cov_2_dect.py
```

### License

This project is under the CC-BY-NC 4.0 license. See [LICENSE](LICENSE) for details.
