import numpy as np
import sys 
from torchvision import transforms
sys.path.append("..")
def out_to_rgb(out,PALETTE,CLASSES):
    palette = np.array(PALETTE)
    assert palette.shape[0] == len(CLASSES)
    assert palette.shape[1] == 3
    assert len(palette.shape) == 2
    color_seg = np.zeros((out.shape[0], out.shape[1], 3), dtype=np.uint8)
    for label, color in enumerate(palette):
        color_seg[out == label, :] = color
    tran=transforms.ToTensor()
    color_seg=tran(color_seg)
    return color_seg
###原始
def out_to_rgb_np(out,PALETTE,CLASSES):
    palette = np.array(PALETTE)
    assert palette.shape[0] == len(CLASSES)
    assert palette.shape[1] == 3
    assert len(palette.shape) == 2
    color_seg = np.zeros((out.shape[0], out.shape[1], 3), dtype=np.uint8)
    for label, color in enumerate(palette):
        color_seg[out == label, :] = color
    return color_seg

# def out_to_rgb_np(out, PALETTE, CLASSES):
#     palette = np.array(PALETTE)
#     assert palette.shape[0] == len(CLASSES)
#     assert palette.shape[1] == 3
#     assert len(palette.shape) == 2

#     if out.ndim == 3 and out.shape[2] == 1:
#         out = np.squeeze(out, axis=2)
#     elif out.ndim == 4 and out.shape[1] == 1:
#         out = np.squeeze(out, axis=1)

#     color_seg = np.zeros((out.shape[0], out.shape[1], 3), dtype=np.uint8)
#     for label, color in enumerate(palette):
#         mask = (out == label)
#         if mask.ndim == 3:
#             mask = mask.any(axis=2)
#         color_seg[mask, :] = color

#     return color_seg


# def out_to_rgb_np(out, PALETTE, CLASSES):
#     palette = np.array(PALETTE)
#     assert palette.shape[0] == len(CLASSES)
#     assert palette.shape[1] == 3
#     assert len(palette.shape) == 2

#     # 检查 out 的维度
#     if out.ndim > 2:
#         out = out[0]  # 取第一个通道

#     # 调试信息
#     print(f"Shape of out in out_to_rgb_np after squeezing: {out.shape}")

#     color_seg = np.zeros((out.shape[0], out.shape[1], 3), dtype=np.uint8)
#     for label, color in enumerate(palette):
#         color_seg[out == label, :] = color
#     return color_seg


