import os.path as osp
import numpy as np
import os
from .convertrgb import out_to_rgb_np
from .palette import CLASSES,PALETTE
from torchvision import transforms
from utils.metrics import *
from PIL import Image
from pyheatmap.heatmap import HeatMap
tran = transforms.ToPILImage()
def saveimages(path,imagelist,label_list,pred_list,imagename_list):
    tran=transforms.ToPILImage()

    for i in range(len(imagelist)):
        image_name=imagename_list[i]
        tmpimage_name=osp.basename(image_name)
        label=label_list[i]
        save_label=out_to_rgb_np(torch.squeeze(label.cpu().detach()),PALETTE,CLASSES)
        save_label=save_label.transpose(1,0,2)
        label=tran(save_label)
        label.save(osp.join(osp.join(path,'label/'),tmpimage_name))

        image_name=imagename_list[i]
        tmpimage_name=osp.basename(image_name)
        pred=pred_list[i]
        save_pred=out_to_rgb_np(pred.cpu().detach(),PALETTE,CLASSES)
        save_pred=save_pred.transpose(1,0,2)
        prediction=tran(save_pred)  
        prediction.save(osp.join(osp.join(path,'prediction/'),tmpimage_name))

        image_name=imagename_list[i]
        tmpimage_name=osp.basename(image_name)
        image=imagelist[i]
        image=image.permute(0,2,1)
        image=tran(image)
        #PIL 分割结果叠加图像显示
        image=image.convert('RGBA')
        prediction=prediction.convert('RGBA')
        image = Image.blend(image, prediction, 0.5)
        tmpimage_name = osp.splitext(tmpimage_name)[0] + '.png'

        image.save(osp.join(osp.join(path,'raw/'),tmpimage_name))


# def saveentropy(path, type, img_list, imagename_list):
#     if not osp.exists(path):
#         os.makedirs(path)
#     if not osp.exists(osp.join(path, type)):
#         os.makedirs(osp.join(path, type))

#     for i in range(len(img_list)):
#         image_name = imagename_list[i]
#         tmpimage_name = osp.basename(image_name)
#         img = img_list[i]
        
#         # Convert tensor to numpy array and squeeze
#         save_img = torch.squeeze(img.cpu().detach()).numpy()
        
#         # Debugging: print the shape of save_img
#         print(f"Shape of save_img: {save_img.shape}")
        
#         # Convert numpy array to PIL image
#         label = Image.fromarray((save_img * 255).astype(np.uint8))
        
#         # Rotate and flip the image
#         label = label.rotate(-90)
#         label = label.transpose(Image.FLIP_LEFT_RIGHT)
        
#         # Save the image
#         label.save(osp.join(osp.join(path, type), tmpimage_name))


# def saveentropy(path, type, img_list, imagename_list):
#     tran=transforms.ToPILImage()
#     if not osp.exists(path):
#         os.makedirs(path)
#     if not osp.exists(osp.join(path, type)):
#         os.makedirs(osp.join(path, type))

#     for i in range(len(img_list)):
#         image_name=imagename_list[i]
#         tmpimage_name=osp.basename(image_name)
#         img = img_list[i]
#         save_img=out_to_rgb_np(torch.squeeze(img.cpu().detach()),PALETTE,CLASSES)
#         save_img=torch.squeeze(img.cpu().detach())
#         save_img=save_img.transpose(1,0,2)
#         label=tran(save_img)
#         label=label.rotate(-90)
#         label=label.transpose(Image.FLIP_LEFT_RIGHT)

#         label.save(osp.join(osp.join(path,type),tmpimage_name))

# def saveentropy(path, type, img_list, imagename_list):
#     tran = transforms.ToPILImage()
#     if not osp.exists(path):
#         os.makedirs(path)
#     if not osp.exists(osp.join(path, type)):
#         os.makedirs(osp.join(path, type))

#     for i in range(len(img_list)):
#         image_name = imagename_list[i]
#         tmpimage_name = osp.basename(image_name)
#         img = img_list[i]
#         save_img = out_to_rgb_np(torch.squeeze(img.cpu().detach()), PALETTE, CLASSES)
#         save_img = torch.squeeze(img.cpu().detach())

#         print(f"Shape of save_img before transpose: {save_img.shape}")
        
#         # 处理 save_img 的形状，确保 PIL 能处理
#         if len(save_img.shape) == 3:
#             # 假设 save_img 的形状是 (channels, height, width)
#             save_img = save_img.permute(1, 2, 0)
        
#         print(f"Shape of save_img after transpose: {save_img.shape}")
        
#         label = Image.fromarray((save_img.numpy() * 255).astype(np.uint8))
#         label = label.rotate(-90)
#         label = label.transpose(Image.FLIP_LEFT_RIGHT)

#         label.save(osp.join(osp.join(path, type), tmpimage_name))

# def saveentropy(path, type, img_list, imagename_list):
#     tran = transforms.ToPILImage()
#     if not osp.exists(path):
#         os.makedirs(path)
#     if not osp.exists(osp.join(path, type)):
#         os.makedirs(osp.join(path, type))

#     for i in range(len(img_list)):
#         image_name = imagename_list[i]
#         tmpimage_name = osp.basename(image_name)
#         img = img_list[i]
#         save_img = out_to_rgb_np(torch.squeeze(img.cpu().detach()).numpy(), PALETTE, CLASSES)

#         print(f"Shape of save_img: {save_img.shape}")

#         label = Image.fromarray((save_img * 255).astype(np.uint8))
#         label = label.rotate(-90)
#         label = label.transpose(Image.FLIP_LEFT_RIGHT)

#         label.save(osp.join(osp.join(path, type), tmpimage_name))


########原始
def saveentropy(path, type, img_list, imagename_list):
    tran=transforms.ToPILImage()
    if not osp.exists(path):
        os.makedirs(path)
    if not osp.exists(osp.join(path, type)):
        os.makedirs(osp.join(path, type))

    for i in range(len(img_list)):
        image_name=imagename_list[i]
        tmpimage_name=osp.basename(image_name)
        img = img_list[i]
        save_img=out_to_rgb_np(torch.squeeze(img.cpu().detach()),PALETTE,CLASSES)
        save_img=torch.squeeze(img.cpu().detach())
        save_img=save_img.transpose(1,0,2)
        label=tran(save_img)
        label=label.rotate(-90)
        label=label.transpose(Image.FLIP_LEFT_RIGHT)

        label.save(osp.join(osp.join(path,type),tmpimage_name))


# def saveentropy(path, type, img_list, imagename_list):
#     tran = transforms.ToPILImage()
#     if not osp.exists(path):
#         os.makedirs(path)
#     if not osp.exists(osp.join(path, type)):
#         os.makedirs(osp.join(path, type))

#     for i in range(len(img_list)):
#         image_name = imagename_list[i]
#         tmpimage_name = osp.basename(image_name)
#         img = img_list[i]

#         # 调试信息
#         print(f"Original shape of img: {img.shape}")

#         # 确保 img 是一个二维张量
#         if img.ndim > 2:
#             img = img[0]  # 取第一个通道

#         # 检查处理后的形状
#         print(f"Shape of img after processing: {img.shape}")

#         save_img = out_to_rgb_np(img.cpu().detach().numpy(), PALETTE, CLASSES)

#         # 检查转换后的形状
#         print(f"Shape of save_img after out_to_rgb_np: {save_img.shape}")

#         # 转换为 PIL 图像
#         label = tran(torch.from_numpy(save_img))
#         label = label.rotate(-90)
#         label = label.transpose(Image.FLIP_LEFT_RIGHT)

#         label.save(osp.join(osp.join(path, type), tmpimage_name))




def save_images_to_one(path,imagelist,label_list,pred_list,entropy_list,imagename_list):

    for i in range(len(imagelist)):
        image_name = imagename_list[i]
        tmpimage_name = osp.basename(image_name)

        image = imagelist[i]
        image=image.permute(0,2,1)
        image=tran(image)

        label = label_list[i]
        save_label = out_to_rgb_np(torch.squeeze(label.cpu().detach()), PALETTE, CLASSES)
        save_label=save_label.transpose(1,0,2)
        label = tran(save_label)

        pred = pred_list[i]
        save_pred = out_to_rgb_np(pred.cpu().detach(), PALETTE, CLASSES)
        save_pred=save_pred.transpose(1,0,2)
        prediction = tran(save_pred)

        entropy = entropy_list[i]
        save_entropy = torch.squeeze(entropy.cpu().detach())
        entropy = tran(save_entropy)
        entropy = entropy.rotate(-90)
        entropy = entropy.transpose(Image.FLIP_LEFT_RIGHT)

        #将4图拼接到一起
        tmpimage_name = osp.splitext(tmpimage_name)[0] + '.png'

        new = Image.new('RGB',(label.size[0] * 2 + 10, label.size[1] * 2 + 10))
        new.paste(image, (0, 0))
        new.paste(label, (0, label.size[1] + 10))
        new.paste(prediction, (label.size[0] + 10, 0))
        new.paste(entropy, (label.size[0] + 10, label.size[1] + 10))

        # save
        new.save(osp.join(path, tmpimage_name))
