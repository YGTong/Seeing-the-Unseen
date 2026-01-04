#!/usr/bin/env python
import argparse
import os
import os.path as osp
import torch
from torch.autograd import Variable
import tqdm
from train import args
from torch.utils.data import DataLoader
from utils.Utils import *
from utils.metrics import SegmentationMetric
import monai
from monai.data import  PILReader
from monai.transforms import *
from train import CLASSES ,PALETTE
from model import DeepLabV3Plus
from utils.data_manager import DeepGlobe
import matplotlib.pyplot as plt
import torch.nn.functional as F


otherargs=args

def enable_dropout(model):
    """ Function to enable the dropout layers during test-time """
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-file', type=str, default="logs/best.pth.tar",
                        help='checkpoint path')
    parser.add_argument( '--gpus', type=list, default=[0])
    parser.add_argument(
        '--datasetdir',
        default='/home/amax/U-Net/Second/MCDropout/dataset',
        help='data root path'
    )
    parser.add_argument(
        '--save-dir',
        type=str,
        default='./results/',
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
        '--batch-size', type=int, default=64, help='batch size for training the model'
    )
    args = parser.parse_args()

    torch.cuda.is_available()
    torch.cuda.device_count()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpus)
    model_file = args.model_file

    if args.mc_samples > 1:
        args.save_dir = osp.join(args.save_dir, 'mc')
    else:
        args.save_dir = osp.join(args.save_dir, 'normal')
# 1. dataset

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
        num_workers=otherargs.num_workers,
        pin_memory=torch.cuda.is_available()
    )
    print(
        f"==> test image numbers: {len(test_files)}"
    )

# 2. model
    model=DeepLabV3Plus(classes=args.num_classes,dropout_rate=args.dropout_rate)
    model=torch.nn.DataParallel(model.cuda(),device_ids=args.gpus)

    if torch.cuda.is_available():
        model = model.cuda()
    print('==> Loading %s model file: %s' %
          (model.__class__.__name__, model_file))
    checkpoint = torch.load(model_file)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    enable_dropout(model)
    print('==> Evaluating with %s' % (otherargs.model_name))

# 3. metric
    seg=SegmentationMetric(otherargs.num_classes)
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
                for i in range(args.mc_samples):
                    prediction = model(data)
                    prediction = torch.softmax(prediction, dim=1)
                    results.append(prediction)
                results = torch.stack(results, dim=0) # mc_samples,batch_size,num_classes,h,w
                entropy = -torch.sum(results * torch.log(results + 1e-12),dim=(0,2))

                pred = torch.mean(results, dim=0) # batch_size,num_classes,h,w
                pred = torch.argmax(pred, dim=1) # batch_size,h,w

                entropy_exp = torch.exp(-entropy)

                #entropy归一化
                # entropy = (entropy - torch.min(entropy)) / (torch.max(entropy) - torch.min(entropy))
                # # new_label = 1/entropy * label + entropy * (1-label)
                # pesudo = (1-entropy) * label + entropy * (1-label)
                # label_ = 1 - label
                # pesudo = 1 - label_ * entropy_exp
                label_ = F.one_hot(label.long(), num_classes=args.num_classes).permute(0, 3, 1, 2).float()

                pseudo_0 = (1 - label) * entropy_exp
                pseudo_1 = 1 - pseudo_0
                pseudo = torch.cat([pseudo_0.unsqueeze(1), pseudo_1.unsqueeze(1)], dim=1)
                # visualize and analysis
                plt.subplot(2,4,1)
                plt.imshow(label_[0,0].cpu().numpy())
                plt.subplot(2,4,2)
                plt.imshow(label_[0,1].cpu().numpy())
                plt.subplot(2,4,3)
                plt.imshow(pseudo[0,0].cpu().numpy())
                plt.subplot(2,4,4)
                plt.imshow(pseudo[0,1].cpu().numpy())
                # visualize the distribution of entropy
                plt.subplot(2,4,5)
                plt.hist(label_[0,0].cpu().numpy().reshape(-1),bins=5)
                plt.subplot(2,4,6)
                plt.hist(label_[0,1].cpu().numpy().reshape(-1),bins=5)
                plt.subplot(2,4,7)
                plt.hist(pseudo[0,0].cpu().numpy().reshape(-1),bins=5)
                plt.subplot(2,4,8)
                plt.hist(pseudo[0,1].cpu().numpy().reshape(-1),bins=5)
                plt.show()

                # if args.save_image:
                #     saveimages(args.save_dir, data, target, pred, img_name)
                #     saveentropy(args.save_dir,'entropy',entropy,img_name)

            else:
                prediction= model(data)
                pred=torch.argmax(torch.softmax(prediction,dim=1),dim=1)

                if args.save_image:
                    saveimages(args.save_dir,data,target,pred,img_name)

            pred,label = pred.cpu().detach().numpy(),label.cpu().detach().numpy()
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
