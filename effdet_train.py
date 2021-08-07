import argparse
import numpy as np
import os
import pandas as pd
import cv2
from matplotlib import pyplot as plt
from tqdm import tqdm
import random

import gc
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SequentialSampler, RandomSampler
from apex import amp

from models import get_effdet
from warmup_scheduler import GradualWarmupScheduler
from dataset import WheatDataset

## uncomment to train with more workers
import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (8192, rlimit[1]))

import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--backbone", default="ed7", type=str,  choices=['ed0', 'ed1', 'ed2', 'ed3', 'ed4', 'ed5', 'ed6', 'ed7'])
parser.add_argument("--img-size", default=768, type=int)
parser.add_argument("--batch-size", default=8, type=int)
parser.add_argument("--workers", default=16, type=int)
parser.add_argument("--warm-epochs", default=20, type=int)
parser.add_argument("--epochs", default=100, type=int)
parser.add_argument("--patience", default=40, type=int)
parser.add_argument("--folds", nargs="+", type=int)
parser.add_argument("--init_lr", default=5e-4, type=float)
parser.add_argument("--warmup-factor", default=10, type=int)
parser.add_argument("--use-amp", default=True, type=lambda x: (str(x).lower() == "true"))
args = parser.parse_args()
print(args)

torch.backends.cudnn.benchmark = True

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def collate_fn(batch):
    return tuple(zip(*batch))

if __name__ == "__main__":
    os.makedirs('checkpoints', exist_ok = True)
    os.makedirs('logs', exist_ok = True)

    df = pd.read_csv('dataset/trainset.csv')
    
    wheat2017_df = pd.read_csv('dataset/wheat2017.csv')
    wheat2017_df = wheat2017_df[['image_id','fold','xmin','ymin','xmax','ymax','isbox','source']].reset_index(drop=True)

    spike_df = pd.read_csv('dataset/spike-wheat.csv')
    spike_df = spike_df[['image_id','fold','xmin','ymin','xmax','ymax','isbox','source']].reset_index(drop=True)

    for fold in args.folds:
        valid_df = df.loc[df['fold'] == fold]
        train_df = df.loc[~df.index.isin(valid_df.index)]
        
        valid_df = valid_df.loc[valid_df['isbox']==True].reset_index(drop=True)
        warm_df = pd.concat([train_df, wheat2017_df, spike_df], ignore_index=True).sample(frac=1).reset_index(drop=True)
        train_df = pd.concat([train_df, wheat2017_df], ignore_index=True).sample(frac=1).reset_index(drop=True)

        warm_dataset = WheatDataset(df=warm_df, img_size=args.img_size, mode='train', network='EffDet')
        train_dataset = WheatDataset(df=train_df, img_size=args.img_size, mode='train', network='EffDet')
        valid_dataset = WheatDataset(df=valid_df, img_size=args.img_size, mode='valid', network='EffDet')
        
        warm_loader = DataLoader(
            warm_dataset,
            batch_size=args.batch_size,
            sampler=RandomSampler(warm_dataset),
            pin_memory=False,
            drop_last=True,
            num_workers=args.workers,
            collate_fn=collate_fn
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            sampler=RandomSampler(train_dataset),
            pin_memory=False,
            drop_last=True,
            num_workers=args.workers,
            collate_fn=collate_fn
        )

        valid_loader = DataLoader(
            valid_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            sampler=SequentialSampler(valid_dataset),
            pin_memory=False,
            num_workers=args.workers,
            collate_fn=collate_fn
        )
        print('WARM: {} | TRAIN: {} | VALID: {}'.format(len(warm_loader.dataset), len(train_loader.dataset), len(valid_loader.dataset)))

        CHECKPOINT = 'checkpoints/effdet_{}_{}_fold{}.pth'.format(args.backbone, args.img_size, fold)
        LOG = 'logs/effdet_{}_{}_fold{}.csv'.format(args.backbone, args.img_size, fold)

        model = get_effdet(args.backbone, num_classes=1, img_size=args.img_size)
        model = model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.init_lr/args.warmup_factor)

        if args.use_amp:
            model, optimizer = amp.initialize(model, optimizer, opt_level='O1')

        scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs-1)
        scheduler = GradualWarmupScheduler(optimizer, multiplier=args.warmup_factor, total_epoch=1, after_scheduler=scheduler_cosine)

        val_loss_min = np.Inf
        if os.path.isfile(LOG):
            os.remove(LOG)
        log_file = open(LOG, 'a')
        log_file.write('Epoch, lr, train_loss, val_loss\n')
        log_file.close()
        
        pat = 0

        loss_hist = AverageMeter()
        for epoch in range(args.epochs):
            scheduler.step(epoch)
            loss_hist.reset()
            model.train()

            if epoch < args.warm_epochs:
                loop = tqdm(warm_loader)
            else:
                loop = tqdm(train_loader)
            for images, targets in loop:
                ### mixup
                if random.random() > 0.5 and epoch >= args.warm_epochs:
                    images = torch.stack(images).cuda()
                    shuffle_indices = torch.randperm(images.size(0))
                    indices = torch.arange(images.size(0))
                    lam = np.clip(np.random.beta(1.0, 1.0), 0.35, 0.65)
                    images = lam * images + (1 - lam) * images[shuffle_indices, :]
                    mix_targets = []
                    for i, si in zip(indices, shuffle_indices):
                        if i.item() == si.item():
                            target = targets[i.item()]
                        else:
                            target = {
                                'boxes': torch.cat([targets[i.item()]['boxes'], targets[si.item()]['boxes']]),
                                'labels': torch.cat([targets[i.item()]['labels'], targets[si.item()]['labels']])
                            }
                        
                        mix_targets.append(target)
                    targets = mix_targets
                else:
                    images = torch.stack(images).cuda()
                boxes = [target['boxes'].cuda().float() for target in targets]
                labels = [target['labels'].cuda().float() for target in targets]

                optimizer.zero_grad()
                with torch.set_grad_enabled(True):
                    loss, _, _ = model(images, boxes, labels)
                    if loss == 0 or not torch.isfinite(loss):
                        continue
                    if args.use_amp:
                        with amp.scale_loss(loss, optimizer) as scaled_loss:
                            scaled_loss.backward()
                    else:
                        loss.backward()
                    loss_hist.update(loss.detach().item(), images.size(0))
                    optimizer.step()
                loop.set_description('Epoch {:03d}/{:03d} | LR: {:.5f}'.format(epoch, args.epochs-1, optimizer.param_groups[0]['lr']))
                loop.set_postfix(loss=loss_hist.avg)
            train_loss = loss_hist.avg

            model.eval()
            loss_hist.reset()
            for images, targets in tqdm(valid_loader):
                images = torch.stack(images).cuda()
                boxes = [target['boxes'].cuda().float() for target in targets]
                labels = [target['labels'].cuda().float() for target in targets]

                with torch.set_grad_enabled(False):
                    loss, _, _ = model(images, boxes, labels)
                    loss_hist.update(loss.detach().item(), images.size(0))
            val_loss = loss_hist.avg

            print('Train loss: {:.5f} | Val loss: {:.5f}'.format(train_loss, val_loss))
            log_file = open(LOG, 'a')
            log_file.write('{}, {:.5f}, {:.5f}, {:.5f}\n'.format(epoch, optimizer.param_groups[0]['lr'], train_loss, val_loss))
            log_file.close()

            if val_loss < val_loss_min:
                print('Valid ap improved from {:.5f} to {:.5f} saving model to {}'.format(val_loss_min, val_loss, CHECKPOINT))
                val_loss_min = val_loss
                pat = 0
                torch.save(model.model.state_dict(), CHECKPOINT)
            else:
                pat += 1

            if pat == args.patience or epoch == args.epochs-1:
                break