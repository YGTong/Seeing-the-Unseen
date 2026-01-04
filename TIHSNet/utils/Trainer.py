
from datetime import datetime
import os
import os.path as osp
import timeit
import numpy as np
import pytz
import torch
import torch.nn.functional as F
# from myloss import myloss
from tensorboardX import SummaryWriter
import tqdm
import socket
from utils.metrics import SegmentationMetric
from utils.Utils import *
from utils.losses import ce_loss

from .convertrgb import out_to_rgb

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
class Trainer(object):

    def __init__(self, cuda, model, optimizer_model, 
                 loader,val_loader,out, max_epoch,classes,palette, stop_epoch=None, lr_gen=1e-3, \
                interval_validate=None, batch_size=8, warmup_epoch=-1):
        self.cuda = cuda
        self.model=model
        self.warmup_epoch = warmup_epoch
        self.optim_model = optimizer_model
        self.lr_gen = lr_gen
        self.batch_size = batch_size
        self.loader=loader
        self.val_loader = val_loader
        self.time_zone = 'Asia/Hong_Kong'
        self.timestamp_start = \
            datetime.now(pytz.timezone(self.time_zone))

        if interval_validate is None:
            self.interval_validate = int(10)
        else:
            self.interval_validate = interval_validate

        self.out = out
        if not osp.exists(self.out):
            os.makedirs(self.out)

        self.val_log_headers = [
            'epoch', 'loss_ce', 'mean_precision', 'mean_recall', 'mean_f1_score', 'mIoU',
            'pa_0', 'recall_0', 'f1_score_0', 'IoU_0', 'pa_1', 'recall_1', 'f1_score_1', 'IoU_1'
        ]

        with open(osp.join(self.out, 'val_log.csv'), 'w') as f:
            f.write(','.join(self.val_log_headers) + '\n')

        self.train_log_headers = ['epoch','iteration', 'loss_ce','loss_kd']

        with open(osp.join(self.out, 'train_log.csv'), 'w') as f:
            f.write(','.join(self.train_log_headers) + '\n')

        log_dir = os.path.join(self.out, 'tensorboard',
                               datetime.now().strftime('%b%d_%H-%M-%S') + '_' + socket.gethostname())
        self.writer = SummaryWriter(log_dir=log_dir)

        self.epoch = 0
        self.iteration = 0
        self.max_epoch = max_epoch
        self.stop_epoch = stop_epoch if stop_epoch is not None else max_epoch

        self.best_IoU = 0.0
        self.best_epoch = -1
        self.lr_scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_model,self.max_epoch,eta_min=1e-6,last_epoch=-1)
        self.palette=palette
        self.classes=classes

    def validate(self):
        # training = self.model.training
        self.model.eval()
        seg=SegmentationMetric(len(self.classes))
        val_ce_loss = 0
        metrics = []
        with torch.no_grad():
            for batch_idx, sample in tqdm.tqdm(
                    enumerate(self.val_loader), total=len(self.val_loader),
                    desc='Valid iteration=%d' % self.iteration, ncols=80,
                    leave=False):
                data = sample['img']
                label=sample['label'].long()
                if self.cuda:
                    data, label= data.cuda(),label.cuda()
                with torch.no_grad():
                    predictions= self.model(data)
                loss_ce = ce_loss(predictions, label)
                loss_data_ce = loss_ce.data.item()
                if np.isnan(loss_data_ce):
                    raise ValueError('loss is nan while validating')
                val_ce_loss += loss_data_ce
                predictions=torch.argmax(torch.softmax(predictions,dim=1),dim=1)
                predictions, label = predictions.cpu().detach().numpy(),label.cpu().detach().numpy()
                predictions, label = predictions.astype(np.int32), label.astype(np.int32)
                _  = seg.addBatch(predictions,label)
           
            val_ce_loss /= len(self.val_loader)
            pa = seg.classPixelAccuracy()
            IoU = seg.IntersectionOverUnion()
            mIoU = seg.meanIntersectionOverUnion()
            recall = seg.recall()

            f1_score=(2 * pa * recall) / (pa  +  recall)
            mean_f1_score=np.mean(f1_score)
            mean_precision=np.mean(pa)
            mean_recall=np.mean(recall)
            print('-'*10,'val','-'*10)
            print(f'val_Pre: {pa[1]:.4f} '
                  f'val_Recall: {recall[1]:.4f} '
                  f'val_F1_score: {f1_score[1]:.4f} '
                  f'val_IoU: {IoU[1]:.4f}')

            self.writer.add_scalar('val_data/loss_CE', val_ce_loss, self.epoch )
            self.writer.add_scalar('val_data/val_mIoU', mIoU, self.epoch )
            self.writer.add_scalar('val_data/val_mPrecision', mean_precision, self.epoch )
            self.writer.add_scalar('val_data/val_mRecall', mean_recall, self.epoch )
            self.writer.add_scalar('val_data/val_mF1-score', mean_f1_score, self.epoch )
            for i in range(len(self.classes)):
                self.writer.add_scalar(('val_Precision/'+self.classes[i]), pa[i], self.epoch )
                self.writer.add_scalar(('val_Recall/'+self.classes[i]), recall[i], self.epoch)
                self.writer.add_scalar(('val_IoU/'+self.classes[i]), IoU[i], self.epoch )
                self.writer.add_scalar(('val_F1_score/'+self.classes[i]), f1_score[i], self.epoch)

            metrics.append(self.epoch)
            metrics.append(val_ce_loss)
            metrics.append(mean_precision)
            metrics.append(mean_recall)
            metrics.append(mean_f1_score)
            metrics.append(mIoU)
            for i in range(len(self.classes)):
                metrics.append(pa[i])
                metrics.append(recall[i])
                metrics.append(f1_score[i])
                metrics.append(IoU[i])

            is_best = IoU[1] > self.best_IoU
            if is_best:
                self.best_epoch = self.epoch + 1
                self.best_IoU = IoU[1]

                torch.save({
                    'epoch': self.epoch,
                    'iteration': self.iteration,
                    'arch': self.model.__class__.__name__,
                    'optim_state_dict': self.optim_model.state_dict(),
                    'model_state_dict': self.model.state_dict(),
                    'learning_rate_gen': get_lr(self.optim_model),
                    'best_IoU': self.best_IoU,
                }, osp.join(self.out, 'best.pth.tar'))

            with open(osp.join(self.out, 'val_log.csv'), 'a') as f:
                log = map(str, metrics)
                f.write(','.join(log) + '\n')
            self.writer.add_scalar('best_model_epoch', self.best_epoch, self.epoch * (len(self.val_loader)))

    def train_epoch(self):

        self.model.train()
        self.running_ce_loss = 0.0
        start_time = timeit.default_timer()
        for batch_idx, sampleS in tqdm.tqdm(
                enumerate(self.loader), total=len(self.loader),
                desc='Train epoch=%d' % self.epoch, ncols=80, leave=False):

            metrics = []

            iteration = batch_idx + self.epoch * len(self.loader)
            self.iteration = iteration
            assert self.model.training

            self.optim_model.zero_grad()

            # 1. train  with random images
            for param in self.model.parameters():
                param.requires_grad = True

            images = sampleS['img'].cuda()
            label=sampleS['label'].cuda().long()

            output= self.model(images)
            loss_ce = ce_loss(output, label)
            prediction=torch.argmax(torch.softmax(output,dim=1),dim=1).float().cpu()

            self.running_ce_loss += loss_ce.item()

            loss_seg_data = loss_ce.data.item()
            loss=loss_ce
            if np.isnan(loss_seg_data):
                raise ValueError('loss is nan while training')
            loss.backward()
            self.optim_model.step()

            # write image log
            if iteration % 10 == 0:  #interval 50 iter writer logs 
                images=images[0, ...].clone().cpu().data.numpy()
                self.writer.add_image('image', images, iteration)           
                label=label[0, ...].clone().cpu().data
                label=out_to_rgb(label,self.palette,self.classes)
                self.writer.add_image('target', label, iteration)
                prediction=prediction[0, ...].clone().cpu().data
                prediction=out_to_rgb(prediction,self.palette,self.classes)
                self.writer.add_image('prediction', prediction, iteration)


            # if self.epoch > self.warmup_epoch:
            # write train different network or freezn backbone parameter
            self.writer.add_scalar('train/loss_seg', loss_seg_data, iteration)
            metrics.append(loss_seg_data)

            with open(osp.join(self.out, 'train_log.csv'), 'a') as f:
                log = [self.epoch, self.iteration] + metrics
                log = map(str, log)
                f.write(','.join(log) + '\n')

        self.running_ce_loss /= len(self.loader)

        stop_time = timeit.default_timer()

        print('\n[Epoch: %d] lr:%f,  Average segLoss: %f, Execution time: %.5f' %
              (self.epoch, get_lr(self.optim_model), self.running_ce_loss, stop_time - start_time))

    def train(self):
        for epoch in tqdm.trange(self.epoch, self.max_epoch,
                                 desc='Train', ncols=80):
            self.epoch = epoch
            self.writer.add_scalar('lr_gen', get_lr(self.optim_model), self.epoch )
            self.train_epoch()
            self.lr_scheduler.step()   
            if (self.epoch+1) % self.interval_validate == 0:
                self.validate()

            if self.stop_epoch == self.epoch:
                print('Stop epoch at %d' % self.stop_epoch)
                break                
        self.writer.close()





