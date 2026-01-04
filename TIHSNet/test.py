#!/usr/bin/env python
import argparse
import os
import os.path as osp
import torch
from torch.autograd import Variable
import tqdm
from torch.utils.data import DataLoader
from utils.Utils import *
from utils.metrics import SegmentationMetric
import monai
from monai.data import  PILReader
from monai.transforms import *
from train import CLASSES ,PALETTE
#from model import DeepLabV3Plus

from utils.data_manager import DeepGlobe
# from model.A2FPN import A2FPN
# from model.exmodel.Unet import UNet
# from model.exmodel.SegNet import SegNet
# from model.MANet import MANet
# from model.A2FPN import A2FPN
# from model.exmodel.dinknet import DinkNet34
# from model.exmodel.ABCNet import ABCNet
# from model.exmodel.FasterNet.NewNet_B import NewUnet3
# from model.UNet.unet import Unet
# from model.UNet.unet import Unet
from model.exmodel.TransUNet.models.networks.TransUnet import get_transNet

# from model.UNet.drouU import drouU
#path_entropy = "/home/amax/U-Net/Second/MCDropout/results/entropy"
def enable_dropout(model):
    """ Function to enable the dropout layers during test-time """
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-file', type=str, default=r"/home/amax/U-Net/third/MCDropout/logs_get_transNetnoloss/best.pth.tar",
                        help='checkpoint path')
    # parser.add_argument( '--gpus', type=str, default=[0])
    parser.add_argument('--gpus', type=str, default='0', help='gpu id')
    parser.add_argument(
        '--datasetdir',
        default='/home/amax/U-Net/Second/MCDropout/dataset',
        help='data root path'
    )
    parser.add_argument(
        '--save-dir',
        type=str,
        default='./results_get_transNetnoloss/',
        help='path to save label',
    )
    parser.add_argument(
        '--dropout-rate', type=float, default=0.1, help='mc dropout rate'
    )
    parser.add_argument(
        '--num-classes', type=int, default=2, help='number of classes'
    )
    parser.add_argument(
        '--save-image', type=bool, default=True, help='save image or not'
    )
    parser.add_argument(
        '--mc-samples', type=int, default=10, help='mc dropout sample times'
    )
    parser.add_argument(
        '--batch-size', type=int, default=4, help='batch size for training the model'
    )
    parser.add_argument(
        '--num-workers', type=int, default=2, help='how many subprocesses to use for dataloading.'
    )


    args = parser.parse_args()

    torch.cuda.is_available()
    torch.cuda.device_count()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpus)

    if args.mc_samples > 1:
        args.save_dir = osp.join(args.save_dir, 'mc')
    else:
        args.save_dir = osp.join(args.save_dir, 'normal')
# 1. dataset
    if not osp.exists(args.save_dir):
        os.makedirs(args.save_dir)

    dataset = DeepGlobe(root = args.datasetdir)
    test_files = dataset.val_files

    test_transforms = Compose(
        [
            LoadImaged(keys=["img", "label"], reader=PILReader, dtype=np.float32),
            AsChannelFirstd(keys=["img"], channel_dim=-1),
            ScaleIntensityd(keys=["img"]),
            # AsDiscreted(keys=['label'], to_onehot=3),
            EnsureTyped(keys=["img", "label"]),
        ]
    )    
    test_dataset = monai.data.Dataset(data=test_files, transform=test_transforms)
    test_loader = DataLoader(
    test_dataset,
        batch_size=args.batch_size,
        shuffle=False, 
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available()
    )
    print(
        f"==> test image numbers: {len(test_files)}"
    )

# 2. model
    args.gpus = [int(x) for x in args.gpus.split(',')]
    # model=drouU(classes=args.num_classes,dropout_rate=args.dropout_rate)
    model=get_transNet(num_classes=2)
    model=torch.nn.DataParallel(model.cuda(),device_ids=args.gpus)

    # args.gpus = [int(x) for x in args.gpus]
    # model=UNet(num_classes=2, in_channels=3)
    # model=torch.nn.DataParallel(model.cuda(),device_ids=args.gpus)

    if torch.cuda.is_available():
        model = model.cuda()
    print('==> Loading %s model file: %s' %
          (model.__class__.__name__, args.model_file))
    checkpoint = torch.load(args.model_file)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    enable_dropout(model)
    print('==> Evaluating with %s' % (args.model_file))

# 3. metric
    seg=SegmentationMetric(args.num_classes)
    with torch.no_grad():
        for batch_idx, (sample) in tqdm.tqdm(enumerate(test_loader),
                                             total=len(test_loader),
                                             ncols=80, leave=False):
            data = sample['img']
            label=sample['label']
            img_mate=sample['img_meta_dict']
            img_name=img_mate['filename_or_obj']
            if torch.cuda.is_available():
                data, label = data.cuda(), label.cuda()
            data, target = Variable(data), Variable(label)
            if args.mc_samples > 1:
                results = []
                logits = []
                for i in range(args.mc_samples):
                    logit = model(data)
                    logits.append(logit)

                    pred = torch.softmax(logit, dim=1)
                    results.append(pred)
                results = torch.stack(results, dim=0) # mc_samples,batch_size,num_classes,h,w
                entropy = -torch.sum(results * torch.log(results + 1e-12),dim=0)

                pred = torch.mean(results, dim=0) # batch_size,num_classes,h,w
                pred = torch.argmax(pred, dim=1) # batch_size,h,w

                #entropy归一化
                # entropy = (entropy - torch.min(entropy)) / (torch.max(entropy) - torch.min(entropy))

                entropy_exp = torch.exp(-entropy)

                if args.save_image:
                    save_images_to_one(args.save_dir, data, target, pred, entropy_exp, img_name)
                    # saveentropy(args.save_dir,'entropy',entropy,img_name)

            else:
                prediction= model(data)
                pred=torch.argmax(torch.softmax(prediction,dim=1),dim=1)

                if args.save_image:
                    saveimages(args.save_dir,data,target,pred,img_name)



            print(f"Shape of pred before numpy conversion: {pred.shape}")
            print(f"Shape of label before numpy conversion: {label.shape}")

            pred,label = pred.cpu().detach().numpy(),label.cpu().detach().numpy()

            print(f"Shape of pred after numpy conversion: {pred.shape}")
            print(f"Shape of label after numpy conversion: {label.shape}")
            
            predictions, label = pred.astype(np.int32), np.squeeze(label.astype(np.int32))
            _  = seg.addBatch(predictions,label)

    pa = seg.classPixelAccuracy()
    IoU = seg.IntersectionOverUnion()
    mIoU = seg.meanIntersectionOverUnion()
    recall = seg.recall()
    f1_score=(2 * pa * recall) / (pa  +  recall)
    mean_f1_score=np.mean(f1_score)
    mean_precision=np.mean(pa)
    mean_recall=np.mean(recall)
    print('''==>mean_Precision : {0}'''.format(mean_precision))
    print('''==>mean_Recall : {0}'''.format(mean_recall))
    print('''==>mean_F1_score : {0}'''.format(mean_f1_score))
    print('''==>mean_IoU : {0}'''.format(mIoU))
    print('''==>Precision : {0}'''.format(pa[1]))
    print('''==>Recall : {0}'''.format(recall[1]))
    print('''==>F1_score : {0}'''.format(f1_score[1]))
    print('''==>IoU : {0}'''.format(IoU[1]))

    with open(osp.join(args.save_dir, 'test_log.csv'), 'a') as f:
        for i in range(len(CLASSES)):
            log1 = [CLASSES[i],'Precision:',pa[i]]
            log2 = ['Recall:',recall[i]]
            log3 = ['IoU:',IoU[i]]
            log4 = ['F1-Score:',f1_score[i]]
            log=log1+log2+log3+log4
            log = map(str, log)
            f.write(','.join(log) + '\n')
        f.write('mean_Precision :'+str(mean_precision)+'\n')
        f.write('mean_Recall :'+str(mean_recall)+'\n')
        f.write('mean_F1_score :'+str(mean_f1_score)+'\n')
        f.write('mean_IoU :'+str(mIoU)+'\n')
if __name__ == '__main__':
    main()
