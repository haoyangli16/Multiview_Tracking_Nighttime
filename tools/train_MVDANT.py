# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import logging
import os
import time
import math
import json
import random
import numpy as np
from pytest import mark

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from torch.nn.utils import clip_grad_norm_
from torch.utils.data.distributed import DistributedSampler

from pysot.utils.lr_scheduler import build_lr_scheduler
from pysot.utils.log_helper import init_log, print_speed, add_file_handler
from pysot.utils.distributed import dist_init, reduce_gradients,\
        average_reduce, get_rank, get_world_size
from pysot.utils.model_load import load_pretrain, restore_from
from pysot.utils.average_meter import AverageMeter
from pysot.utils.misc import describe, commit
from pysot.models.model_builder import ModelBuilder
from pysot.models.trans_discriminator import TransformerDiscriminator
from pysot.datasets.dataset import TrkDataset
from pysot.core.config import cfg
import torch.nn.functional as F
from pysot.utils.loss_con import calculate_the_loss_con

os.environ['CUDA_VISIBLE_DEVICES'] = '0,2'

import sys

sys.path.append(os.getcwd())

logger = logging.getLogger('global')
parser = argparse.ArgumentParser(description='siamcar tracking')
parser.add_argument('--cfg', type=str, default=os.getcwd() + '/experiments/siamcar_r50/config_MUFAT.yaml',
                    help='configuration of tracking')
parser.add_argument('--seed', type=int, default=123456,
                    help='random seed')
parser.add_argument('--local_rank', type=int, default=0,
                    help='compulsory for pytorch launcer')
args = parser.parse_args()


def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def build_data_loader(domain):
    logger.info("build train dataset")
    # train_dataset
    train_dataset = TrkDataset(domain)
    logger.info("build dataset done")

    train_sampler = None
    if get_world_size() > 1:
        train_sampler = DistributedSampler(train_dataset)
    train_loader = DataLoader(train_dataset,
                              batch_size=cfg.TRAIN.BATCH_SIZE,
                              num_workers=cfg.TRAIN.NUM_WORKERS,
                              pin_memory=True,
                              sampler=train_sampler)
    return train_loader


def build_opt_lr(model, current_epoch=0):
    if current_epoch >= cfg.BACKBONE.TRAIN_EPOCH:
        for layer in cfg.BACKBONE.TRAIN_LAYERS:
            for param in getattr(model.backbone, layer).parameters():
                param.requires_grad = True
            for m in getattr(model.backbone, layer).modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.train()
    else:
        for param in model.backbone.parameters():
            param.requires_grad = False
        for m in model.backbone.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    trainable_params = []
    trainable_params += [{'params': filter(lambda x: x.requires_grad,
                                           model.backbone.parameters()),
                          'lr': cfg.BACKBONE.LAYERS_LR * cfg.TRAIN.BASE_LR}]
    
    # 添加不同Neck和Tracker的参数
    if cfg.ADJUST.ADJUST:
        trainable_params += [{'params': model.neck.parameters(),
                              'lr': cfg.TRAIN.BASE_LR}]
    if cfg.ALIGN.ALIGN:
        trainable_params += [{'params': model.align.parameters(),
                              'lr': cfg.TRAIN.BASE_LR_d}]

    trainable_params += [{'params': model.car_head.parameters(),
                          'lr': cfg.TRAIN.BASE_LR}]
    
    trainable_params += [{'params': model.down.parameters(),
                          'lr': cfg.TRAIN.BASE_LR}]
    optimizer = torch.optim.SGD(trainable_params,
                                momentum=cfg.TRAIN.MOMENTUM,
                                weight_decay=cfg.TRAIN.WEIGHT_DECAY)

    lr_scheduler = build_lr_scheduler(optimizer, epochs=cfg.TRAIN.EPOCH)
    lr_scheduler.step(cfg.TRAIN.START_EPOCH)
    return optimizer, lr_scheduler


def log_grads(model, tb_writer, tb_index):
    def weights_grads(model):
        grad = {}
        weights = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad[name] = param.grad
                weights[name] = param.data
        return grad, weights

    grad, weights = weights_grads(model)
    feature_norm, car_norm = 0, 0
    for k, g in grad.items():
        _norm = g.data.norm(2)
        weight = weights[k]
        w_norm = weight.norm(2)
        if 'feature' in k:
            feature_norm += _norm ** 2
        else:
            car_norm += _norm ** 2

        tb_writer.add_scalar('grad_all/'+k.replace('.', '/'),
                             _norm, tb_index)
        tb_writer.add_scalar('weight_all/'+k.replace('.', '/'),
                             w_norm, tb_index)
        tb_writer.add_scalar('w-g/'+k.replace('.', '/'),
                             w_norm/(1e-20 + _norm), tb_index)
    tot_norm = feature_norm + car_norm
    tot_norm = tot_norm ** 0.5
    feature_norm = feature_norm ** 0.5
    car_norm = car_norm ** 0.5

    tb_writer.add_scalar('grad/tot', tot_norm, tb_index)
    tb_writer.add_scalar('grad/feature', feature_norm, tb_index)
    tb_writer.add_scalar('grad/car', car_norm, tb_index)

def weightedMSE(D_out, label):
    # D_label = torch.FloatTensor(D_out.data.size()).fill_(1).cuda() * label.unsqueeze(1).unsqueeze(2).unsqueeze(3).cuda()
    # D_label = torch.FloatTensor(D_out.data.size()).fill_(label).cuda()
    return torch.mean((D_out - label.cuda()).abs() ** 2)

def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))

def adjust_learning_rate_D(args, optimizer, i_iter):
    lr = lr_poly(args.TRAIN.BASE_LR_d, i_iter, args.TRAIN.EPOCH, 0.8)
    for k in optimizer.param_groups:
        k['lr'] = lr
    # optimizer.param_groups[0]['lr'] = lr
    # if len(optimizer.param_groups) > 1:
    #     optimizer.param_groups[1]['lr'] = lr * 10
    return lr

def train(source_loader, source_loader_2, target_loader, model, model_2, optimizer,optimizer_2, lr_scheduler, tb_writer, Disc, Disc_2, optimizer_D,optimizer_D_2):
    cur_lr = lr_scheduler.get_cur_lr()
    cur_lr_d = adjust_learning_rate_D(cfg, optimizer_D, cfg.TRAIN.START_EPOCH)

    rank = get_rank()

    average_meter = AverageMeter()

    def is_valid_number(x):
        return not(math.isnan(x) or math.isinf(x) or x > 1e4)

    world_size = get_world_size()
    num_per_epoch = min(len(target_loader.dataset),len(source_loader.dataset),len(source_loader_2.dataset)) // \
        cfg.TRAIN.EPOCH // (cfg.TRAIN.BATCH_SIZE * world_size)
    start_epoch = cfg.TRAIN.START_EPOCH
    epoch = start_epoch

    if not os.path.exists(cfg.TRAIN.SNAPSHOT_DIR) and \
            get_rank() == 0:
        os.makedirs(cfg.TRAIN.SNAPSHOT_DIR)

    logger.info("model\n{}".format(describe(model.module)))
    end = time.time()

    # 0: daytime ?
    # 1: nighttime ?
    source_label = 0
    target_label = 1
    
    target_data = enumerate(target_loader)
    source_data = enumerate(source_loader)
    source_data_2 = enumerate(source_loader_2)

    # for idx in range(cfg.TRAIN.EPOCH * num_per_epoch):
    for idx, data in enumerate(target_loader):
        # data = target_data.__next__()[1]
        if epoch != idx // num_per_epoch + start_epoch:
            epoch = idx // num_per_epoch + start_epoch

            if get_rank() == 0:
                torch.save(
                        {'epoch': epoch,
                         'state_dict': model.module.state_dict(),
                         'optimizer': optimizer.state_dict()},
                        cfg.TRAIN.SNAPSHOT_DIR+'/checkpoint_e%d.pth' % (epoch))
                torch.save(
                        {'epoch': epoch,
                         'state_dict': model_2.module.state_dict(),
                         'optimizer': optimizer_2.state_dict()},
                        cfg.TRAIN.SNAPSHOT_DIR+'/checkpoint_2_e%d.pth' % (epoch))
                '''
                torch.save( # save discriminator
                        {'epoch': epoch,
                         'state_dict': Disc.module.state_dict(),
                         'optimizer': optimizer_D.state_dict()},
                        cfg.TRAIN.SNAPSHOT_DIR+'/d_checkpoint_e%d.pth' % (epoch))
                
                torch.save( # save discriminator2
                        {'epoch': epoch,
                         'state_dict': Disc_2.module.state_dict(),
                         'optimizer': optimizer_D_2.state_dict()},
                        cfg.TRAIN.SNAPSHOT_DIR+'/d2_checkpoint_e%d.pth' % (epoch))
                '''
            if epoch == cfg.TRAIN.EPOCH:
                return

            if cfg.BACKBONE.TRAIN_EPOCH == epoch:
                logger.info('start training backbone.')
                optimizer, lr_scheduler = build_opt_lr(model.module, epoch)
                logger.info("model\n{}".format(describe(model.module)))\

                optimizer_2, lr_scheduler_2 = build_opt_lr(model_2.module, epoch)
                logger.info("model_2\n{}".format(describe(model_2.module)))

            lr_scheduler.step(epoch)
            cur_lr = lr_scheduler.get_cur_lr()
            cur_lr_d = adjust_learning_rate_D(cfg, optimizer_D, epoch)
            logger.info('epoch: {}'.format(epoch+1))

        tb_idx = idx
        if idx % num_per_epoch == 0 and idx != 0:
            for idx, pg in enumerate(optimizer.param_groups):
                logger.info('epoch {} lr {}'.format(epoch+1, pg['lr']))
                if rank == 0:
                    tb_writer.add_scalar('lr/group{}'.format(idx+1),
                                         pg['lr'], tb_idx)

        data_time = average_reduce(time.time() - end)
        if rank == 0:
            tb_writer.add_scalar('time/data', data_time, tb_idx)

        '''train G
        '''
        # ---Loss_adv---
        for param in Disc.parameters():
            param.requires_grad = False # 屏蔽判别器
        for param in Disc_2.parameters():
            param.requires_grad = False # 屏蔽判别器
            
        # i) 与source_1进行特征对齐
        outputs, zf, xf = model(data, target=0) # target_data
        interp = nn.Upsample(size=(128, 128), mode='bilinear', align_corners=True)
        zf_up_t = [interp(_zf) for _zf in zf]
        xf_up_t = [interp(_xf) for _xf in xf]
        D_out_z = torch.stack([Disc(_zf_up_t) for _zf_up_t in zf_up_t]).sum(0) / 3.
        D_out_x = torch.stack([Disc(_xf_up_t) for _xf_up_t in xf_up_t]).sum(0) / 3.

        # Create target labels for multiple classes
        batch_size = D_out_z.size(0)
        target_labels = torch.zeros(batch_size, 4).to(device)

        source1_label = target_labels
        for i in range(batch_size):
            source1_label[i, 1] = 1

        # Compute the cross-entropy loss between the predicted labels and target labels
        loss_adv = 0.1 * (F.cross_entropy(D_out_z, torch.argmax(source1_label, dim=1)) +
                        F.cross_entropy(D_out_x, torch.argmax(source1_label, dim=1)))


        
        loss_gt_t = outputs['total_loss'].mean()

        loss_ADV = loss_adv + loss_gt_t

        if is_valid_number(loss_ADV.data.item()):
            optimizer.zero_grad()
            optimizer_D.zero_grad()
            optimizer_D_2.zero_grad()
            loss_ADV.backward()

        # ii) 与source_2进行特征对齐
        outputs, zf, xf = model_2(data, target=0) # target_data
        interp = nn.Upsample(size=(128, 128), mode='bilinear', align_corners=True)
        zf_up_t_2 = [interp(_zf) for _zf in zf]
        xf_up_t_2 = [interp(_xf) for _xf in xf]
        D_out_z_2 = torch.stack([Disc_2(_zf_up_t) for _zf_up_t in zf_up_t_2]).sum(0) / 3.
        D_out_x_2 = torch.stack([Disc_2(_xf_up_t) for _xf_up_t in xf_up_t_2]).sum(0) / 3.

        source2_label = target_labels
        for i in range(batch_size):
            source2_label[i, 2] = 1

        # Compute the cross-entropy loss between the predicted labels and target labels
        loss_adv_2 = 0.1 * (F.cross_entropy(D_out_z_2, torch.argmax(source2_label, dim=1)) +
                        F.cross_entropy(D_out_x_2, torch.argmax(source2_label, dim=1)))

        loss_gt_t_2 = outputs['total_loss'].mean()

        loss_ADV2 = loss_adv_2 + loss_gt_t_2

        if is_valid_number(loss_ADV2.data.item()):
            loss_ADV2.backward()

        # ---Loss_GT---
        data = source_data.__next__()[1]
        data_2 = source_data_2.__next__()[1]
        
        # i) source_1
        outputs, zf, xf = model(data, target=0) # source_data
        interp = nn.Upsample(size=(128, 128), mode='bilinear', align_corners=True)
        zf_up_s = [interp(_zf) for _zf in zf]
        xf_up_s = [interp(_xf) for _xf in xf]
        loss_gt = outputs['total_loss'].mean()
        
        if is_valid_number(loss_gt.data.item()):
            loss_gt.backward()
        
        # ii) source_2
        outputs_2, zf2, xf2 = model_2(data_2, target=0) # source_data
        interp = nn.Upsample(size=(128, 128), mode='bilinear', align_corners=True)
        zf_up_s_2 = [interp(_zf) for _zf in zf2]
        xf_up_s_2 = [interp(_xf) for _xf in xf2]
        loss_gt_2 = outputs_2['total_loss'].mean()
        
        if is_valid_number(loss_gt_2.data.item()):
            loss_gt_2.backward()
        
        # ---Loss_con---
        data = target_data.__next__()[1]
        cls1, loc1, cen1 = model(data, target=1)
        cls2, loc2, cen2 = model_2(data, target=1)
        outputs_con = calculate_the_loss_con(cls1, loc1, cen1, cls2, loc2, cen2)
        loss_con = outputs_con['total_loss_con'].mean()
        
        if is_valid_number(loss_con.data.item()):
            loss_con.backward()
    
        
        '''train D
        '''
        # ---Loss_D---
        loss_train_adv = 0
        loss_train_adv_1 = 0
        loss_train_adv_2 = 0
        for param in Disc.parameters():
            param.requires_grad = True 
        for param in Disc_2.parameters():
            param.requires_grad = True 
         
        zf_up_t = [_zf_up_t.detach() for _zf_up_t in zf_up_t]
        xf_up_t = [_xf_up_t.detach() for _xf_up_t in xf_up_t]
        zf_up_t_2 = [_zf_up_t.detach() for _zf_up_t in zf_up_t_2]
        xf_up_t_2 = [_xf_up_t.detach() for _xf_up_t in xf_up_t_2]
        
        # 1. target
        # i) target_1
        D_out_1_t= torch.stack([Disc(_zf_up_t) for _zf_up_t in zf_up_t]).sum(0) / 3.
        D_out_2_t = torch.stack([Disc(_xf_up_t) for _xf_up_t in xf_up_t]).sum(0) / 3.
        D_target_label = torch.FloatTensor(D_out_z.data.size()).fill_(target_label)
        loss_d = 0.1*weightedMSE(D_out_1_t, D_target_label) + 0.1*weightedMSE(D_out_2_t, D_target_label)
        if is_valid_number(loss_d.data.item()):
            loss_d.backward() 
        loss_train_adv += loss_d.item()
        loss_train_adv_1 += loss_d.item()
        
        # ii) target_2
        D_out_1_t= torch.stack([Disc_2(_zf_up_t) for _zf_up_t in zf_up_t_2]).sum(0) / 3.
        D_out_2_t = torch.stack([Disc_2(_xf_up_t) for _xf_up_t in xf_up_t_2]).sum(0) / 3.
        D_target_label = torch.FloatTensor(D_out_z_2.data.size()).fill_(target_label)
        loss_d = (0.1*weightedMSE(D_out_1_t, D_target_label) + 0.1*weightedMSE(D_out_2_t, D_target_label))
        if is_valid_number(loss_d.data.item()):
            loss_d.backward() 
        loss_train_adv += loss_d.item()
        loss_train_adv_2 += loss_d.item()
        # 2. source
        # i) source_1
        zf_up_s = [_zf_up_s.detach() for _zf_up_s in zf_up_s]
        xf_up_s = [_xf_up_s.detach() for _xf_up_s in xf_up_s]
        D_out_1_s = torch.stack([Disc(_zf_up_s) for _zf_up_s in zf_up_s]).sum(0) / 3.
        D_out_2_s = torch.stack([Disc(_xf_up_s) for _xf_up_s in xf_up_s]).sum(0) / 3.
        D_source_label = torch.FloatTensor(D_out_z.data.size()).fill_(source_label)
        loss_d = 0.1*weightedMSE(D_out_1_s, D_source_label) + 0.1*weightedMSE(D_out_2_s, D_source_label)
        if is_valid_number(loss_d.data.item()):
            loss_d.backward() 
        loss_train_adv += loss_d.item()
        loss_train_adv_1 += loss_d.item()
        # ii) source_2
        zf_up_s_2 = [_zf_up_s.detach() for _zf_up_s in zf_up_s_2]
        xf_up_s_2 = [_xf_up_s.detach() for _xf_up_s in xf_up_s_2]
        D_out_1_s = torch.stack([Disc_2(_zf_up_s) for _zf_up_s in zf_up_s_2]).sum(0) / 3.
        D_out_2_s= torch.stack([Disc_2(_xf_up_s) for _xf_up_s in xf_up_s_2]).sum(0) / 3.
        D_source_label = torch.FloatTensor(D_out_z_2.data.size()).fill_(source_label)
        loss_d = (0.1*weightedMSE(D_out_1_s, D_source_label) + 0.1*weightedMSE(D_out_2_s, D_source_label))
        if is_valid_number(loss_d.data.item()):
            loss_d.backward() 
        loss_train_adv += loss_d.item()
        loss_train_adv_2 += loss_d.item()
        
        if is_valid_number(loss_gt.data.item()):
            reduce_gradients(model)
            reduce_gradients(model_2)
            reduce_gradients(Disc)
            reduce_gradients(Disc_2)
            if rank == 0 and cfg.TRAIN.LOG_GRADS:
                log_grads(model.module, tb_writer, tb_idx)
                log_grads(model_2.module, tb_writer, tb_idx)

            # clip gradient
            clip_grad_norm_(model.parameters(), cfg.TRAIN.GRAD_CLIP)
            clip_grad_norm_(model_2.parameters(), cfg.TRAIN.GRAD_CLIP)
            clip_grad_norm_(Disc.parameters(), cfg.TRAIN.GRAD_CLIP)
            clip_grad_norm_(Disc_2.parameters(),cfg.TRAIN.GRAD_CLIP)
            optimizer.step()
            optimizer_2.step()
            optimizer_D.step()
            optimizer_D_2.step()

        batch_time = time.time() - end
        batch_info = {}
        batch_info['batch_time'] = average_reduce(batch_time)
        batch_info['data_time'] = average_reduce(data_time)
        batch_info['loss_fool'] = average_reduce(loss_adv)
        batch_info['loss_fool_2'] = average_reduce(loss_adv_2)
        # batch_info['loss_con'] = average_reduce(loss_con)
        batch_info['loss_train_adv'] = average_reduce(loss_train_adv)
        batch_info['loss_train_adv_1'] = average_reduce(loss_train_adv_1)
        batch_info['loss_train_adv_2'] = average_reduce(loss_train_adv_2)
        # batch_info['loss_d'] = average_reduce(loss_d)
        

        for k, v in sorted(outputs.items()):
            batch_info[k] = average_reduce(v.mean().data.item())
            
        for k, v in sorted(outputs_2.items()):
            t=k+'_2'
            batch_info[t] = average_reduce(v.mean().data.item())

        for k, v in sorted(outputs_con.items()):
            batch_info[k] = average_reduce(v.mean().data.item())
        average_meter.update(**batch_info)

        if rank == 0:
            for k, v in batch_info.items():
                tb_writer.add_scalar(k, v, tb_idx)

            if (idx+1) % cfg.TRAIN.PRINT_FREQ == 0:
                info = "Epoch: [{}][{}/{}] lr: {:.6f}\n".format(
                            epoch+1, (idx+1) % num_per_epoch,
                            num_per_epoch, cur_lr)
                for cc, (k, v) in enumerate(batch_info.items()):
                    if cc % 2 == 0:
                        info += ("\t{:s}\t").format(
                                getattr(average_meter, k))
                    else:
                        info += ("{:s}\n").format(
                                getattr(average_meter, k))
                logger.info(info)
                print_speed(idx+1+start_epoch*num_per_epoch,
                            average_meter.batch_time.avg,
                            cfg.TRAIN.EPOCH * num_per_epoch)
        end = time.time()


def main():
    rank, world_size = dist_init()
    # rank = 0
    logger.info("init done")

    # load cfg
    cfg.merge_from_file(args.cfg)
    if rank == 0:
        if not os.path.exists(cfg.TRAIN.LOG_DIR):
            os.makedirs(cfg.TRAIN.LOG_DIR)
        init_log('global', logging.INFO)
        if cfg.TRAIN.LOG_DIR:
            add_file_handler('global',
                             os.path.join(cfg.TRAIN.LOG_DIR, 'logs.txt'),
                             logging.INFO)

        logger.info("Version Information: \n{}\n".format(commit()))
        logger.info("config \n{}".format(json.dumps(cfg, indent=4)))

    # create model
    model = ModelBuilder().train()
    dist_model = nn.DataParallel(model).cuda()
    
    model_2 = ModelBuilder().train()
    dist_model_2 = nn.DataParallel(model_2).cuda()

    # load pretrained backbone weights
    if cfg.BACKBONE.PRETRAINED:
        cur_path = os.path.dirname(os.path.realpath(__file__))
        print("cur_path: ", cur_path)
        print("backbone_path: ", backbone_path)
        backbone_path = os.path.join(cur_path, '../', cfg.BACKBONE.PRETRAINED)
        # print(backbone_path)
        load_pretrain(model.backbone, backbone_path)
        load_pretrain(model_2.backbone, backbone_path)

    # create tensorboard writer
    if rank == 0 and cfg.TRAIN.LOG_DIR:
        tb_writer = SummaryWriter(cfg.TRAIN.LOG_DIR)
    else:
        tb_writer = None

    # build dataset loader
    # 1个目标域; 2个源域
    target_loader = build_data_loader('target')
    source_loader = build_data_loader('source')
    source_loader_2 = build_data_loader('source_2')

    # build optimizer and lr_scheduler
    optimizer, lr_scheduler = build_opt_lr(dist_model.module,
                                           cfg.TRAIN.START_EPOCH)
    optimizer_2, lr_scheduler_2 = build_opt_lr(dist_model_2.module,
                                           cfg.TRAIN.START_EPOCH)
    # create Discriminator鉴别器
    # D1
    model_Disc = TransformerDiscriminator(channels=256) # 特征的通道数
    model_Disc.train()
    model_Disc.cuda()
    dist_Disc = nn.DataParallel(model_Disc)
    optimizer_D = torch.optim.Adam(model_Disc.parameters(), lr=cfg.TRAIN.BASE_LR_d, betas=(0.9, 0.99)) # TODO 写到cfg里
    optimizer_D.zero_grad()
    
    # D2
    model_Disc_2 = TransformerDiscriminator(channels=256) # 特征的通道数
    model_Disc_2.train()
    model_Disc_2.cuda()
    dist_Disc_2 = nn.DataParallel(model_Disc_2)
    optimizer_D_2 = torch.optim.Adam(model_Disc_2.parameters(), lr=cfg.TRAIN.BASE_LR_d, betas=(0.9, 0.99)) # TODO 写到cfg里
    optimizer_D_2.zero_grad()

    # resume training
    if cfg.TRAIN.RESUME:
        logger.info("resume from {}".format(cfg.TRAIN.RESUME))
        assert os.path.isfile(cfg.TRAIN.RESUME), \
            '{} is not a valid file.'.format(cfg.TRAIN.RESUME)
        model, optimizer, cfg.TRAIN.START_EPOCH = \
            restore_from(model, optimizer, cfg.TRAIN.RESUME)
    # load pretrain
    elif cfg.TRAIN.PRETRAINED:
        load_pretrain(model, cfg.TRAIN.PRETRAINED)
    
    # resume training
    if cfg.TRAIN.RESUME_2:
        logger.info("resume from {}".format(cfg.TRAIN.RESUME_2))
        assert os.path.isfile(cfg.TRAIN.RESUME), \
            '{} is not a valid file.'.format(cfg.TRAIN.RESUME_2)
        model_2, optimizer_2, cfg.TRAIN.START_EPOCH = \
            restore_from(model_2, optimizer_2, cfg.TRAIN.RESUME_2)
    # load pretrain
    elif cfg.TRAIN.PRETRAINED_2:
        load_pretrain(model_2, cfg.TRAIN.PRETRAINED_2)

    if cfg.TRAIN.RESUME_D:
        logger.info("resume from {}".format(cfg.TRAIN.RESUME_D))
        assert os.path.isfile(cfg.TRAIN.RESUME_D), \
            '{} is not a valid file.'.format(cfg.TRAIN.RESUME_D)
        model_Disc, optimizer_D, cfg.TRAIN.START_EPOCH = \
            restore_from(model_Disc, optimizer_D, cfg.TRAIN.RESUME_D)
            
    if cfg.TRAIN.RESUME_D_2:
        logger.info("resume from {}".format(cfg.TRAIN.RESUME_D_2))
        assert os.path.isfile(cfg.TRAIN.RESUME_D_2), \
            '{} is not a valid file.'.format(cfg.TRAIN.RESUME_D_2)
        model_Disc_2, optimizer_D_2, cfg.TRAIN.START_EPOCH = \
            restore_from(model_Disc_2, optimizer_D_2, cfg.TRAIN.RESUME_D_2)

    dist_model = nn.DataParallel(model)
    dist_model_2 = nn.DataParallel(model_2)
    dist_Disc = nn.DataParallel(model_Disc)
    dist_Disc_2 = nn.DataParallel(model_Disc_2)

    logger.info(lr_scheduler)
    logger.info("model prepare done")

    # start training
    train(source_loader, source_loader_2, target_loader, dist_model,dist_model_2, optimizer, optimizer_2, lr_scheduler, tb_writer, dist_Disc, dist_Disc_2, optimizer_D, optimizer_D_2)


if __name__ == '__main__':
    seed_torch(args.seed)
    main()
