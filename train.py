import os
import os.path as osp
import time
import math
from datetime import timedelta
from argparse import ArgumentParser

import numpy as np
import random
import torch.backends.cudnn as cudnn

import torch
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from tqdm import tqdm

from east_dataset import EASTDataset
from dataset import SceneTextDataset, ValidSceneTextDataset
from model import EAST

import wandb
from importlib import import_module
from config import Config

import numpy as np
import random
from detect import get_bboxes
from deteval import calc_deteval_metrics

def parse_args():
    parser = ArgumentParser()

    # Conventional args
    parser.add_argument('--data_dir', type=str, default=Config.data_dir)
                        # default=os.environ.get('SM_CHANNEL_TRAIN', '../input/data/ICDAR17_Korean'))
    parser.add_argument('--use_val', type=bool, default=Config.use_val)
    parser.add_argument('--val_dir', type=str, default=Config.val_dir)

    parser.add_argument('--model_dir', type=str, default=Config.model_dir)
                        # default=os.environ.get('SM_MODEL_DIR','trained_models'))

    parser.add_argument('--device', default=Config.device)
                        # default='cuda' if cuda.is_available() else 'cpu')
    parser.add_argument('--num_workers', type=int, default=Config.num_workers)

    parser.add_argument('--image_size', type=int, default=Config.image_size)
    parser.add_argument('--input_size', type=int, default=Config.input_size)
    parser.add_argument('--batch_size', type=int, default=Config.batch_size)
    parser.add_argument('--learning_rate', type=float, default=Config.learning_rate)
    parser.add_argument('--max_epoch', type=int, default=Config.max_epoch)
    parser.add_argument('--save_interval', type=int, default=Config.save_interval)

    parser.add_argument('--optimizer', type=str, default=Config.optimizer)
    parser.add_argument('--early_stopping', type=str, default=Config.early_stopping)
    parser.add_argument('--expr_name', type=str, default=Config.expr_name)
    parser.add_argument('--resume_from', type=str, default=Config.resume_from)
    parser.add_argument('--save_point', nargs="+", type=int, default=Config.save_point)

    parser.add_argument('--seed', type=int, default=Config.seed)
    
    args = parser.parse_args()

    if args.input_size % 32 != 0:
        raise ValueError('`input_size` must be a multiple of 32')

    return args

def set_seed(seed) :
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    #torch.cuda.manual_seed_all(seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    print(f"seed : {seed}")


def do_training(data_dir, use_val, val_dir, model_dir, device, image_size, input_size, num_workers, batch_size,
                learning_rate, max_epoch, save_interval, optimizer, early_stopping, expr_name, resume_from, save_point, seed):
    
    set_seed(seed)
    
    dataset = SceneTextDataset(data_dir, split='train', image_size=image_size, crop_size=input_size)
    dataset = EASTDataset(dataset)
    num_batches = math.ceil(len(dataset) / batch_size)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    
    if use_val:
        #using training data of ICDAR17
        val_dataset = ValidSceneTextDataset(val_dir, split='train', image_size=image_size, crop_size=input_size, color_jitter=False)
        val_dataset.load_image()
        print(f"Load valid data {len(val_dataset)}")
        valid_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=ValidSceneTextDataset.collate_fn)
        val_num_batches = math.ceil(len(val_dataset) / batch_size)
        max_f1 = 0.
    else:
        min_loss = 10000.
        
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    start_epoch = 0
    model = EAST()
    model.to(device)
    opt_module = getattr(import_module("torch.optim"), optimizer)  # default: Adam
    optimizer = opt_module(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate
    )
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[max_epoch // 2], gamma=0.1)
    
    if resume_from:         # 이어서 학습
        model_data = torch.load(resume_from)
        model.load_state_dict(model_data['model_state_dict'])
        model.to(device)
        optimizer.load_state_dict(model_data['optimizer_state_dict'])
        start_epoch = model_data['epoch']

    
    model.train()
    er_cnt=0
    for epoch in range(start_epoch, max_epoch):
        epoch_loss, epoch_start = 0, time.time()
        with tqdm(total=num_batches) as pbar:
            for img, gt_score_map, gt_geo_map, roi_mask in train_loader:
                pbar.set_description('[Epoch {}]'.format(epoch + 1))

                extra_info = None
                output = model.train_step(img, gt_score_map, gt_geo_map, roi_mask)
                if isinstance(output, tuple):
                    loss, extra_info = output
                else:
                    loss = output
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_val = loss.item()
                epoch_loss += loss_val

                pbar.update(1)
                val_dict = dict()
                if extra_info:
                    val_dict = {
                        'Cls loss': extra_info['cls_loss'], 'Angle loss': extra_info['angle_loss'],
                        'IoU loss': extra_info['iou_loss']
                    }
                else:
                    val_dict = {
                        'Cls loss': None, 'Angle loss': None,
                        'IoU loss': None
                    }
                wandb.log(val_dict)
                pbar.set_postfix(val_dict)
        scheduler.step()
        wandb.log({"Mean loss": epoch_loss / num_batches, "epoch": epoch})
        print('Mean loss: {:.4f} | Elapsed time: {}'.format(
            epoch_loss / num_batches, timedelta(seconds=time.time() - epoch_start)))

        if not use_val:
            if isinstance(epoch_loss / num_batches, float):
                if min_loss > epoch_loss / num_batches:
                    er_cnt=0
                    
                    min_loss = epoch_loss / num_batches
                    print('Best Mean loss: {:.4f}'.format(min_loss))
                    if not osp.exists(model_dir):
                        os.makedirs(model_dir)

                    ckpt_fpath = osp.join(model_dir, 'best_mean_loss.pth')

                    print(f'Best model saved at epoch{epoch+1}!')
                    torch.save({
                        'epoch': epoch,
                        'optimizer_state_dict': optimizer.state_dict(),
                        'model_state_dict': model.state_dict()},
                        ckpt_fpath)
                else:
                    er_cnt += 1
                    if er_cnt >= early_stopping:

                        print(f'early stopping at epoch {epoch+1}')

                        break

        if (epoch + 1) % save_interval == 0:
            if not osp.exists(model_dir):
                os.makedirs(model_dir)

            ckpt_fpath = osp.join(model_dir, 'real-latest.pth')

            torch.save({
                'epoch': epoch,
                'optimizer_state_dict': optimizer.state_dict(),
                'model_state_dict': model.state_dict()},
                ckpt_fpath)

            print(f'latest model saved at epoch{epoch+1}')
            
            
        if (epoch + 1) in save_point:
            if not osp.exists(model_dir):
                os.makedirs(model_dir)

            ckpt_fpath = osp.join(model_dir, f'epoch-{epoch+1}.pth')

            torch.save({
                'epoch': epoch,
                'optimizer_state_dict': optimizer.state_dict(),
                'model_state_dict': model.state_dict()},
                ckpt_fpath)

            print(f'model saved at epoch{epoch+1}')
        
        if use_val:
            print("validation start!")
            val_epoch_loss = 0
            val_cls_loss = 0
            val_angle_loss = 0
            val_iou_loss = 0            
            pred_bboxes_dict = dict()
            gt_bboxes_dict = dict()
            transcriptions_dict = dict()
            with tqdm(total=val_num_batches) as pbar:
                with torch.no_grad():
                    model.eval()
                    for step, (img, gt_score_map, gt_geo_map, roi_mask, vertices, orig_sizes, labels, transcriptions, fnames) in enumerate(valid_loader):
                        pbar.set_description('[Valid {}]'.format(epoch + 1))

                        loss, extra_info = model.train_step(img, gt_score_map, gt_geo_map, roi_mask)

                        score_maps, geo_maps = extra_info['score_map'], extra_info['geo_map']
                        score_maps, geo_maps = score_maps.cpu().numpy(), geo_maps.cpu().numpy()

                        by_sample_bboxes = []
                        for i, (score_map, geo_map, orig_size, vertice, transcription, fname) in enumerate(zip(score_maps, geo_maps, orig_sizes, vertices, transcriptions, fnames)):
                            map_margin = int(abs(orig_size[0] - orig_size[1]) * 0.25 * image_size / max(orig_size))
                            if orig_size[0] > orig_size[1]:
                                score_map, geo_map = score_map[:, :, :-map_margin], geo_map[:, :, :-map_margin]
                            else:
                                score_map, geo_map = score_map[:, :-map_margin, :], geo_map[:, :-map_margin, :]

                            bboxes = get_bboxes(score_map, geo_map)
                            if bboxes is None:
                                bboxes = np.zeros((0, 4, 2), dtype=np.float32)
                            else:
                                bboxes = bboxes[:, :8].reshape(-1, 4, 2)

                            pred_bboxes_dict[fname] = bboxes
                            gt_bboxes_dict[fname] = vertice
                            transcriptions_dict[fname] = transcription

                        loss_val = loss.item()
                        if loss_val is not None:
                            val_epoch_loss += loss_val

                        pbar.update(1)
                        val_dict = {
                            'Cls loss': extra_info['cls_loss'],
                            'Angle loss': extra_info['angle_loss'],
                            'IoU loss': extra_info['iou_loss']
                        }
                        pbar.set_postfix(val_dict)

                        if val_dict['Cls loss'] is not None:
                            val_cls_loss += val_dict['Cls loss']
                            val_angle_loss += val_dict['Angle loss']
                            val_iou_loss += val_dict['IoU loss']
            resDict = calc_deteval_metrics(pred_bboxes_dict, gt_bboxes_dict, transcriptions_dict)

            print('[Valid {}]: f1_score : {:.4f} | precision : {:.4f} | recall : {:.4f}'.format(
                    epoch+1, resDict['total']['hmean'], resDict['total']['precision'], resDict['total']['recall']))

            wandb.log({ "val/loss": val_epoch_loss / val_num_batches,
                    "val/cls_loss": val_cls_loss / val_num_batches,
                    "val/angle_loss": val_angle_loss / val_num_batches,
                    "val/iou_loss": val_iou_loss / val_num_batches,
                    "val/recall": resDict['total']['recall'],
                    "val/precision": resDict['total']['precision'],
                    "val/f1_score": resDict['total']['hmean'],
                    "epoch":epoch+1})
            
            val_f1 = resDict['total']['hmean']
            
            if max_f1 < val_f1:
                er_cnt=0
    
                max_f1 = val_f1
                print('Best f1: {:.4f}'.format(val_f1))
                if not osp.exists(model_dir):
                    os.makedirs(model_dir)
    
                ckpt_fpath = osp.join(model_dir, 'best_f1.pth')
    
                print(f'Best model saved at epoch{epoch+1}!')
                torch.save({
                    'epoch': epoch,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'model_state_dict': model.state_dict()},
                    ckpt_fpath)
            else:
                er_cnt += 1
                if er_cnt >= early_stopping:
                    print(f'early stopping at epoch {epoch+1}')
                    break            

def main(args):
    do_training(**args.__dict__)


if __name__ == '__main__':
    args = parse_args()
    wandb.init(project="data-annotation", name=args.expr_name, entity="level2-cv-16")
    wandb.config.update(args)
    main(args)
