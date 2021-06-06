import torch
import torch.nn as nn
import torch.nn.functional as F
from .basic import *

__all__ = ['cross_stitch', 'cross_stitch_multicase']


class CrossStitchDeepLabv3(nn.Module):
    def __init__(self, tasks):
        super(CrossStitchDeepLabv3, self).__init__()
        self.tasks = tasks
        # self.backbones = torch.nn.ModuleList([ResnetDilated(resnet18(pretrained=False)) for _ in range(len(self.tasks))])
        self.backbones = torch.nn.ModuleList(
            [ResnetDilated(resnet34(pretrained=False)) for _ in range(len(self.tasks))])
        self.cross_stitch_units = torch.nn.Parameter(torch.ones(4, len(self.tasks)), requires_grad=True)
        # ch = [256, 512, 1024, 2048]

        self.shared_convs = torch.nn.ModuleList([nn.Sequential(self.backbones[_].conv1, self.backbones[_].bn1,
                                                               self.backbones[_].relu1, self.backbones[_].maxpool)
                                                 for _ in range(len(self.tasks))])

        # Define task-specific decoders using ASPP modules
        self.decoders = nn.ModuleList()
        for task in tasks:
            if task == 'segment_semantic':
                output_channels = 18
            elif task == 'depth_zbuffer':
                output_channels = 1
            elif task == 'normal':
                output_channels = 3
            elif task == 'edge_occlusion':
                output_channels = 1
            elif task == 'reshading':
                output_channels = 1
            elif task == 'keypoints2d':
                output_channels = 1
            elif task == 'edge_texture':
                output_channels = 1
            elif task == 'rgb':
                output_channels = 3
            elif task == 'principal_curvature':
                output_channels = 2
            else:
                raise NotImplementedError('Unknown task', task)
            # 2048 for resnet50, 512 for resnet18
            self.decoders.append(DeepLabHead(512, output_channels))

    def forward(self, x):
        x_s = [self.shared_convs[_](x) for _ in range(len(self.tasks))]

        x_s = [self.backbones[_].layer1(x_s[_]) for _ in range(len(self.tasks))]
        x_s = [x_s[_] * self.cross_stitch_units[0][_] for _ in range(len(self.tasks))]

        x_s = [self.backbones[_].layer2(x_s[_]) for _ in range(len(self.tasks))]
        x_s = [x_s[_] * self.cross_stitch_units[1][_] for _ in range(len(self.tasks))]

        x_s = [self.backbones[_].layer3(x_s[_]) for _ in range(len(self.tasks))]
        x_s = [x_s[_] * self.cross_stitch_units[2][_] for _ in range(len(self.tasks))]

        x_s = [self.backbones[_].layer4(x_s[_]) for _ in range(len(self.tasks))]
        x_s = [x_s[_] * self.cross_stitch_units[3][_] for _ in range(len(self.tasks))]

        # Task specific decoders
        output = {}
        for i in range(len(self.tasks)):
            ret = F.interpolate(self.decoders[i](x_s[i]), size=256, mode='bilinear', align_corners=True)
            output[self.tasks[i]] = ret

        return output


class CrossStitchDeepLabv3MultiCase(nn.Module):
    def __init__(self, tasks):
        super(CrossStitchDeepLabv3MultiCase, self).__init__()
        self.tasks = tasks
        # self.backbones = torch.nn.ModuleList([ResnetDilated(resnet18(pretrained=False)) for _ in range(len(self.tasks))])
        self.backbones = torch.nn.ModuleList(
            [ResnetDilated(resnet34(pretrained=False)) for _ in range(len(self.tasks))])
        self.cross_stitch_units = torch.nn.Parameter(torch.ones(4, len(self.tasks)), requires_grad=True)
        # ch = [256, 512, 1024, 2048]

        self.shared_convs = torch.nn.ModuleList([nn.Sequential(self.backbones[_].conv1, self.backbones[_].bn1,
                                                               self.backbones[_].relu1, self.backbones[_].maxpool)
                                                 for _ in range(len(self.tasks))])

        # Define task-specific decoders using ASPP modules
        self.decoders = nn.ModuleList()
        for task in tasks:
            temp_task = task.split('/')[-1]
            if temp_task == 'segment_semantic':
                output_channels = 18
            elif temp_task == 'depth_zbuffer':
                output_channels = 1
            elif temp_task == 'normal':
                output_channels = 3
            elif temp_task == 'edge_occlusion':
                output_channels = 1
            elif temp_task == 'reshading':
                output_channels = 1
            elif temp_task == 'keypoints2d':
                output_channels = 1
            elif temp_task == 'edge_texture':
                output_channels = 1
            elif temp_task == 'rgb':
                output_channels = 3
            elif temp_task == 'principal_curvature':
                output_channels = 2
            else:
                raise NotImplementedError('Unknown task', temp_task)
            # 2048 for resnet50, 512 for resnet18
            self.decoders.append(DeepLabHead(512, output_channels))

    def forward(self, x):
        x_s = [self.shared_convs[_](x[self.tasks[_]]) for _ in range(len(self.tasks))]

        x_s = [self.backbones[_].layer1(x_s[_]) for _ in range(len(self.tasks))]
        x_s = [x_s[_] * self.cross_stitch_units[0][_] for _ in range(len(self.tasks))]

        x_s = [self.backbones[_].layer2(x_s[_]) for _ in range(len(self.tasks))]
        x_s = [x_s[_] * self.cross_stitch_units[1][_] for _ in range(len(self.tasks))]

        x_s = [self.backbones[_].layer3(x_s[_]) for _ in range(len(self.tasks))]
        x_s = [x_s[_] * self.cross_stitch_units[2][_] for _ in range(len(self.tasks))]

        x_s = [self.backbones[_].layer4(x_s[_]) for _ in range(len(self.tasks))]
        x_s = [x_s[_] * self.cross_stitch_units[3][_] for _ in range(len(self.tasks))]

        # Task specific decoders
        output = {}
        for i in range(len(self.tasks)):
            ret = F.interpolate(self.decoders[i](x_s[i]), size=256, mode='bilinear', align_corners=True)
            output[self.tasks[i]] = ret

        return output


def cross_stitch(tasks):
    return CrossStitchDeepLabv3(tasks)


def cross_stitch_multicase(tasks):
    return CrossStitchDeepLabv3MultiCase(tasks)