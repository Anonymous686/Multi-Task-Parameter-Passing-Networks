import os
import time
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
sll, nll, dll = 0, 0, 0


def get_seg_loss(dataset, args):
    if dataset == 'NYU_v2':
        seg_num_class = 40
    elif dataset == 'CityScapes':
        seg_num_class = 19
    else:
        raise NotImplementedError()

    def seg_loss(output, target):
        global sll
        target = F.interpolate(target.float(), size=output.shape[-2:]).permute(0, 2, 3, 1).contiguous().view(-1)
        output = output.permute(0, 2, 3, 1).contiguous().view(-1, seg_num_class)
        sll = nn.CrossEntropyLoss(ignore_index=255 if dataset == 'NYU_v2' else -1)(output, target.long())
        sll = sll * args.loss_weight['s']
        return sll

    return seg_loss


def get_sn_loss(dataset, args):

    def sn_loss(output, target):
        global nll
        target = F.interpolate(target.float(), size=output.shape[-2:]).permute(0, 2, 3, 1).contiguous().view(-1, 3)
        labels = (target.max(dim=1)[0] < 255)
        output = output.permute(0, 2, 3, 1).contiguous().view(-1, 3)
        output = output[labels]
        target = target[labels]
        output = F.normalize(output)
        target = F.normalize(target)
        nll = 1 - nn.CosineSimilarity()(output, target).mean()
        # balance loss
        # nll = nll * 20
        nll = nll * args.loss_weight['n']
        return nll

    return sn_loss


def get_depth_loss(dataset, args):

    def depth_loss(output, target):
        global dll
        target = F.interpolate(target.float(), size=output.shape[-2:])
        binary_mask = (torch.sum(target, dim=1) > 3 * 1e-5).unsqueeze(1)
        output = output.masked_select(binary_mask)
        target = target.masked_select(binary_mask)
        dll = nn.L1Loss()(output, target)
        # balance loss
        # for nyu_v2
        # dll = dll * 3
        # for cityscapes
        dll = dll * args.loss_weight['d']
        # dll = dll * 5
        return dll

    return depth_loss


def get_baseline_loss(losses):

    def baseline_loss(output, target):
        sum_loss = None
        num = 0
        for n, t in target.items():
            if n in losses:
                o = output[n].float()
                this_loss = losses[n](o, t)
                num += 1
                if sum_loss is not None:
                    sum_loss = sum_loss + this_loss
                else:
                    sum_loss = this_loss
        return sum_loss

    return baseline_loss


def get_baseline_losses_and_tasks(args):
    task_str = args.tasks
    losses = {}
    criteria = {}
    tasks = []
    if 'nyu_v2' in args.data_dir:
        dataset = 'NYU_v2'
    elif 'cityscapes' in args.data_dir:
        dataset = 'CityScapes'
    else:
        raise NotImplementedError('Unknown dataset calling loss generator', args.data_dir)

    if 's' in task_str:
        losses['segment_semantic'] = get_seg_loss(dataset, args)
        criteria['ss_l'] = lambda x, y: sll
        tasks.append('segment_semantic')
    if 'd' in task_str:
        losses['depth_zbuffer'] = get_depth_loss(dataset, args)
        criteria['depth_l'] = lambda x, y: dll
        tasks.append('depth_zbuffer')
    if 'n' in task_str:
        losses['normal'] = get_sn_loss(dataset, args)
        criteria['norm_l'] = lambda x, y: nll
        tasks.append('normal')

    baseline_loss = get_baseline_loss(losses)
    return baseline_loss, losses, criteria, tasks