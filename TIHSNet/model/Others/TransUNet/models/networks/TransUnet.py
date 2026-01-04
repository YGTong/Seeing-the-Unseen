import torch
import torch.nn as nn
import functools
import torch.nn.functional as F
from model.exmodel.TransUNet.models.networks.vit_seg_modeling import VisionTransformer as ViT_seg
from model.exmodel.TransUNet.models.networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg

def get_transNet(num_classes):
    img_size = 512
    vit_patches_size = 16
    vit_name = 'R50-ViT-B_16'

    config_vit = CONFIGS_ViT_seg[vit_name]
    config_vit.n_classes = num_classes
    config_vit.n_skip = 3
    if vit_name.find('R50') != -1:
        config_vit.patches.grid = (int(img_size / vit_patches_size), int(img_size / vit_patches_size))
    net = ViT_seg(config_vit, img_size=img_size, num_classes=num_classes)
    return net


if __name__ == '__main__':
    net = get_transNet(2)
    img = torch.randn((2, 3, 512, 512))
    segments = net(img)
    print(net )
    # for edge in edges:
    #     print(edge.size())
