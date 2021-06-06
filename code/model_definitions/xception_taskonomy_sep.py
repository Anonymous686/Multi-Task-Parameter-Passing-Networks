import math
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import torch
import numpy as np
from .ozan_rep_fun import ozan_rep_function, trevor_rep_function, OzanRepFunction, TrevorRepFunction

__all__ = ['xception_taskonomy_sep', 'xception_taskonomy_jointlearn', 'XceptionTaskonomySep', 'XceptionTaskonomyBase',
           'xception_multitask', 'xception_taskonomy_jointlearn_better', 'xception_multitask_multicase', 'XceptionTaskonomySepSimple']


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False,
                 groupsize=1):
        super(SeparableConv2d, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation,
                               groups=max(1, in_channels // groupsize), bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


class Block(nn.Module):
    def __init__(self, in_filters, out_filters, reps, strides=1, start_with_relu=True, grow_first=True):
        super(Block, self).__init__()

        if out_filters != in_filters or strides != 1:
            self.skip = nn.Conv2d(in_filters, out_filters, 1, stride=strides, bias=False)
            self.skipbn = nn.BatchNorm2d(out_filters)
        else:
            self.skip = None

        self.relu = nn.ReLU(inplace=True)
        rep = []

        filters = in_filters
        if grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters, out_filters, 3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(out_filters))
            filters = out_filters

        for i in range(reps - 1):
            rep.append(self.relu)
            rep.append(SeparableConv2d(filters, filters, 3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(filters))

        if not grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters, out_filters, 3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(out_filters))
            filters = out_filters

        if not start_with_relu:
            rep = rep[1:]
        else:
            rep[0] = nn.ReLU(inplace=False)

        if strides != 1:
            # rep.append(nn.AvgPool2d(3,strides,1))
            rep.append(nn.Conv2d(filters, filters, 2, 2))
        self.rep = nn.Sequential(*rep)

    def forward(self, inp):
        x = self.rep(inp)

        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp
        x += skip
        return x


class Encoder(nn.Module):
    def __init__(self, sizes=None):
        super(Encoder, self).__init__()
        if sizes is None:
            sizes = [32, 64, 128, 256, 728, 728, 728, 728, 728, 728, 728, 728, 728]
        self.conv1 = nn.Conv2d(3, sizes[0], 3, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(sizes[0])
        self.relu = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=False)

        self.conv2 = nn.Conv2d(sizes[0], sizes[1], 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(sizes[1])
        # do relu here

        self.block1 = Block(sizes[1], sizes[2], 2, 2, start_with_relu=False, grow_first=True)
        self.block2 = Block(sizes[2], sizes[3], 2, 2, start_with_relu=True, grow_first=True)
        self.block3 = Block(sizes[3], sizes[4], 2, 2, start_with_relu=True, grow_first=True)

        self.block4 = Block(sizes[4], sizes[5], 3, 1, start_with_relu=True, grow_first=True)
        self.block5 = Block(sizes[5], sizes[6], 3, 1, start_with_relu=True, grow_first=True)
        self.block6 = Block(sizes[6], sizes[7], 3, 1, start_with_relu=True, grow_first=True)
        self.block7 = Block(sizes[7], sizes[8], 3, 1, start_with_relu=True, grow_first=True)

        self.block8 = Block(sizes[8], sizes[9], 3, 1, start_with_relu=True, grow_first=True)
        self.block9 = Block(sizes[9], sizes[10], 3, 1, start_with_relu=True, grow_first=True)
        self.block10 = Block(sizes[10], sizes[11], 3, 1, start_with_relu=True, grow_first=True)
        self.block11 = Block(sizes[11], sizes[12], 3, 1, start_with_relu=True, grow_first=True)

    def forward(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)

        representation = self.relu2(x)

        return representation


def interpolate(inp, size):
    t = inp.type()
    inp = inp.float()
    out = nn.functional.interpolate(inp, size=size, mode='bilinear', align_corners=False)
    if out.type() != t:
        out = out.half()
    return out


class Decoder(nn.Module):
    def __init__(self, output_channels=32, num_classes=None, half_sized_output=False, small_decoder=True):
        super(Decoder, self).__init__()

        self.output_channels = output_channels
        self.num_classes = num_classes
        self.half_sized_output = half_sized_output
        self.relu = nn.ReLU(inplace=True)
        if num_classes is not None:
            self.block12 = Block(728, 1024, 2, 2, start_with_relu=True, grow_first=False)

            self.conv3 = SeparableConv2d(1024, 1536, 3, 1, 1)
            self.bn3 = nn.BatchNorm2d(1536)

            # do relu here
            self.conv4 = SeparableConv2d(1536, 2048, 3, 1, 1)
            self.bn4 = nn.BatchNorm2d(2048)

            self.fc = nn.Linear(2048, num_classes)
        else:
            if small_decoder:
                self.upconv1 = nn.ConvTranspose2d(512, 128, 2, 2)
                self.bn_upconv1 = nn.BatchNorm2d(128)
                self.conv_decode1 = nn.Conv2d(128, 128, 3, padding=1)
                self.bn_decode1 = nn.BatchNorm2d(128)
                self.upconv2 = nn.ConvTranspose2d(128, 64, 2, 2)
                self.bn_upconv2 = nn.BatchNorm2d(64)
                self.conv_decode2 = nn.Conv2d(64, 64, 3, padding=1)
                self.bn_decode2 = nn.BatchNorm2d(64)
                self.upconv3 = nn.ConvTranspose2d(64, 48, 2, 2)
                self.bn_upconv3 = nn.BatchNorm2d(48)
                self.conv_decode3 = nn.Conv2d(48, 48, 3, padding=1)
                self.bn_decode3 = nn.BatchNorm2d(48)
                if half_sized_output:
                    self.upconv4 = nn.Identity()
                    self.bn_upconv4 = nn.Identity()
                    self.conv_decode4 = nn.Conv2d(48, output_channels, 3, padding=1)
                else:
                    self.upconv4 = nn.ConvTranspose2d(48, 32, 2, 2)
                    self.bn_upconv4 = nn.BatchNorm2d(32)
                    self.conv_decode4 = nn.Conv2d(32, output_channels, 3, padding=1)
            else:
                self.upconv1 = nn.ConvTranspose2d(512, 256, 2, 2)
                self.bn_upconv1 = nn.BatchNorm2d(256)
                self.conv_decode1 = nn.Conv2d(256, 256, 3, padding=1)
                self.bn_decode1 = nn.BatchNorm2d(256)
                self.upconv2 = nn.ConvTranspose2d(256, 128, 2, 2)
                self.bn_upconv2 = nn.BatchNorm2d(128)
                self.conv_decode2 = nn.Conv2d(128, 128, 3, padding=1)
                self.bn_decode2 = nn.BatchNorm2d(128)
                self.upconv3 = nn.ConvTranspose2d(128, 96, 2, 2)
                self.bn_upconv3 = nn.BatchNorm2d(96)
                self.conv_decode3 = nn.Conv2d(96, 96, 3, padding=1)
                self.bn_decode3 = nn.BatchNorm2d(96)
                if half_sized_output:
                    self.upconv4 = nn.Identity()
                    self.bn_upconv4 = nn.Identity()
                    self.conv_decode4 = nn.Conv2d(96, output_channels, 3, padding=1)
                else:
                    self.upconv4 = nn.ConvTranspose2d(96, 64, 2, 2)
                    self.bn_upconv4 = nn.BatchNorm2d(64)
                    self.conv_decode4 = nn.Conv2d(64, output_channels, 3, padding=1)

    def forward(self, representation):
        if self.num_classes is None:
            x = self.upconv1(representation)
            x = self.bn_upconv1(x)
            x = self.relu(x)
            x = self.conv_decode1(x)
            x = self.bn_decode1(x)
            x = self.relu(x)
            x = self.upconv2(x)
            x = self.bn_upconv2(x)
            x = self.relu(x)
            x = self.conv_decode2(x)

            x = self.bn_decode2(x)
            x = self.relu(x)
            x = self.upconv3(x)
            x = self.bn_upconv3(x)
            x = self.relu(x)
            x = self.conv_decode3(x)
            x = self.bn_decode3(x)
            x = self.relu(x)
            if not self.half_sized_output:
                x = self.upconv4(x)
                x = self.bn_upconv4(x)
                x = self.relu(x)
            x = self.conv_decode4(x)

        else:
            x = self.block12(representation)

            x = self.conv3(x)
            x = self.bn3(x)
            x = self.relu(x)

            x = self.conv4(x)
            x = self.bn4(x)
            x = self.relu(x)

            x = F.adaptive_avg_pool2d(x, (1, 1))
            x = x.view(x.size(0), -1)
            x = self.fc(x)
        return x


class XceptionTaskonomySep(nn.Module):
    """
    Xception optimized for the ImageNet dataset, as specified in
    https://arxiv.org/pdf/1610.02357.pdf
    """

    def __init__(self, task: str, size=1, ozan=False):
        """ Constructor
        Args:
        """
        super(XceptionTaskonomySep, self).__init__()
        assert type(task) == str
        sizes = [32, 64, 128, 256, 728, 728, 728, 728, 728, 728, 728, 728, 728]
        assert size == 1
        if size == 1:
            sizes = [32, 64, 128, 256, 728, 728, 728, 728, 728, 728, 728, 728, 728]
        elif size == .2:
            sizes = [16, 32, 64, 256, 320, 320, 320, 320, 320, 320, 320, 320, 320]
        elif size == .3:
            sizes = [32, 64, 128, 256, 728, 728, 728, 728, 728, 728, 728, 728, 728]
        elif size == .4:
            sizes = [32, 64, 128, 256, 728, 728, 728, 728, 728, 728, 728, 728, 728]
        elif size == .5:
            sizes = [24, 48, 96, 192, 512, 512, 512, 512, 512, 512, 512, 512, 512]
        elif size == .8:
            sizes = [32, 64, 128, 248, 648, 648, 648, 648, 648, 648, 648, 648, 648]
        elif size == 2:
            sizes = [32, 64, 128, 256, 728, 728, 728, 728, 728, 728, 728, 728, 728]
        elif size == 4:
            sizes = [64, 128, 256, 512, 1456, 1456, 1456, 1456, 1456, 1456, 1456, 1456, 1456]

        self.encoder = Encoder(sizes=sizes)
        pre_rep_size = sizes[-1]

        self.task = task
        self.ozan = ozan

        self.final_conv = SeparableConv2d(pre_rep_size, 512, 3, 1, 1)
        self.final_conv_bn = nn.BatchNorm2d(512)

        temp_task = task.split('/')[-1]

        if temp_task == 'segment_semantic':
            output_channels = 18
        elif temp_task == 'depth_zbuffer':
            output_channels = 1
        elif temp_task == 'normal':
            output_channels = 3
        elif temp_task == 'edge_occlusion':
            output_channels = 1
        elif temp_task == 'keypoints2d':
            output_channels = 1
        elif temp_task == 'edge_texture':
            output_channels = 1
        elif temp_task == 'reshading':
            output_channels = 1
        elif temp_task == 'rgb':
            output_channels = 3
        elif temp_task == 'principal_curvature':
            output_channels = 2
        else:
            raise NotImplementedError('Unknown task', temp_task)

        self.decoder = Decoder(output_channels)

        # ------- init weights --------
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        # -----------------------------


    def forward(self, input):
        rep = self.encoder(input)

        rep = self.final_conv(rep)
        rep = self.final_conv_bn(rep)

        # if self.ozan:
        #     OzanRepFunction.n = 1
        #     rep = ozan_rep_function(rep)
        #     rep = rep[0]
        # else:
        #     TrevorRepFunction.n = 1
        #     rep = trevor_rep_function(rep)

        output = self.decoder(rep)

        return output



class XceptionTaskonomySepSimple(nn.Module):
    """
    Xception optimized for the ImageNet dataset, as specified in
    https://arxiv.org/pdf/1610.02357.pdf
    """

    def __init__(self, task: str, size=1, ozan=False, num_seg_cls=19, **kwargs):
        """ Constructor
        Args:
        """
        super(XceptionTaskonomySepSimple, self).__init__()
        assert type(task) == str
        sizes = [32, 64, 128, 256, 728, 728, 728, 728, 728, 728, 728, 728, 728]
        assert size == 1
        if size == 1:
            sizes = [32, 64, 128, 256, 728, 728, 728, 728, 728, 728, 728, 728, 728]
        elif size == .2:
            sizes = [16, 32, 64, 256, 320, 320, 320, 320, 320, 320, 320, 320, 320]
        elif size == .3:
            sizes = [32, 64, 128, 256, 728, 728, 728, 728, 728, 728, 728, 728, 728]
        elif size == .4:
            sizes = [32, 64, 128, 256, 728, 728, 728, 728, 728, 728, 728, 728, 728]
        elif size == .5:
            sizes = [24, 48, 96, 192, 512, 512, 512, 512, 512, 512, 512, 512, 512]
        elif size == .8:
            sizes = [32, 64, 128, 248, 648, 648, 648, 648, 648, 648, 648, 648, 648]
        elif size == 2:
            sizes = [32, 64, 128, 256, 728, 728, 728, 728, 728, 728, 728, 728, 728]
        elif size == 4:
            sizes = [64, 128, 256, 512, 1456, 1456, 1456, 1456, 1456, 1456, 1456, 1456, 1456]

        self.encoder = Encoder(sizes=sizes)
        pre_rep_size = sizes[-1]

        self.task = task
        self.ozan = ozan

        self.final_conv = SeparableConv2d(pre_rep_size, 512, 3, 1, 1)
        self.final_conv_bn = nn.BatchNorm2d(512)

        temp_task = task.split('/')[-1]

        if temp_task == 'segment_semantic':
            output_channels = num_seg_cls
        elif temp_task == 'depth_zbuffer':
            output_channels = 1
        elif temp_task == 'normal':
            output_channels = 3
        elif temp_task == 'edge_occlusion':
            output_channels = 1
        elif temp_task == 'keypoints2d':
            output_channels = 1
        elif temp_task == 'edge_texture':
            output_channels = 1
        elif temp_task == 'reshading':
            output_channels = 1
        elif temp_task == 'rgb':
            output_channels = 3
        elif temp_task == 'principal_curvature':
            output_channels = 2
        else:
            raise NotImplementedError('Unknown task', temp_task)

        self.decoder = Decoder(output_channels)

        # ------- init weights --------
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        # -----------------------------


    def forward(self, input):
        rep = self.encoder(input)
        rep = self.final_conv(rep)
        rep = self.final_conv_bn(rep)
        output = self.decoder(rep)

        return output


class XceptionTaskonomyJointLearn(XceptionTaskonomySep):
    def __init__(self, tasks: list, size=1, ozan=False):
        super(XceptionTaskonomyJointLearn, self).__init__(tasks[0], size, ozan)
        self.model_num = len(tasks)
        self.models = nn.ModuleList([XceptionTaskonomySep(tasks[i], size, ozan) for i in range(self.model_num)])
        self.fake_models = nn.ModuleList([XceptionTaskonomySep(tasks[i], size, ozan) for i in range(self.model_num)])
        self.n_workers = self.model_num
        # init W
        # W starts from uniform distribution
        # self.W = torch.nn.Parameter(torch.tensor(np.ones((self.n_workers, self.n_workers)) * 1.0), requires_grad=True)
        # W starts from self-loop
        self.W = torch.nn.Parameter(torch.tensor(np.eye(self.n_workers) * 1.0), requires_grad=True)

        self.co_ef = None

        self.names_to_pop = [name for name, para in self.named_parameters() if
                             not (name.startswith('W') or name.startswith('models'))]
        self.names_to_pop_buffer = [name for name, para in self.named_buffers() if
                                    not (name.startswith('W') or name.startswith('models'))]

        self.names_to_aggregate = [name for name, para in self.named_parameters() if
                                   not (name.startswith('W') or name.startswith('models') or name.startswith(
                                       'fake') or name.startswith('decoder'))]

        self.names_to_aggregate_buffer = [name for name, para in self.named_buffers() if
                                          not (name.startswith('W') or name.startswith('models') or name.startswith(
                                              'fake') or name.startswith('decoder'))]
        # print(self.names_to_aggregate_buffer)

        # for name in self.names_to_pop:
        #     trace = name.split('.')
        #     self.pop_params(trace, self)

        # for name in self.names_to_pop_buffer:
        #     trace = name.split('.')
        #     self.pop_buffers(trace, self)
        # for name in self.names_to_aggregate:
        #     print(name, end='|')
        #     print(self.get_params(name.split('.'), self.models[0]).shape)
        # exit(0)

    def set_buffer(self):
        for name in self.names_to_pop:
            trace = name.split('.')
            self.pop_params(trace, self)

        for name in self.names_to_pop_buffer:
            trace = name.split('.')
            self.pop_buffers(trace, self)
        # here!!!
        for name in self.names_to_aggregate_buffer:
            trace = name.split('.')
            for i in range(self.model_num):
                self.set_params(trace, self.fake_models[i], self.get_params(trace, self.models[i]))

    def update_co_ef(self):
        self.co_ef = self.get_co_ef()

    def get_co_ef(self):
        return self.W / self.W.sum(dim=-1, keepdim=True).expand_as(self.W)
        # return self.W + torch.eye(self.W.shape[0]).cuda()

    @staticmethod
    def get_params(trace, target_model):
        temp = target_model
        for _ in range(len(trace)):
            temp = temp.__getattr__(trace[_])
        return temp

    @staticmethod
    def set_params(trace, target_model, param):
        temp = target_model
        for _ in range(len(trace) - 1):
            temp = temp.__getattr__(trace[_])
        temp.__setattr__(trace[-1], param)

    @staticmethod
    def pop_params(trace, target_model):
        temp = target_model
        for _ in range(len(trace) - 1):
            temp = temp.__getattr__(trace[_])
        temp._parameters.pop(trace[-1])

    @staticmethod
    def pop_buffers(trace, target_model):
        temp = target_model
        for _ in range(len(trace) - 1):
            temp = temp.__getattr__(trace[_])
        temp._buffers.pop(trace[-1])

    def aggregate(self, task_index):
        for name in self.names_to_aggregate:
            trace = name.split('.')
            aggr_param = sum(
                [co_ef * self.get_params(trace, sub_model) for co_ef, sub_model in
                 zip(self.co_ef[task_index], self.models)])
            self.set_params(trace, self.fake_models[task_index], aggr_param)

    def aggregate_as_state_dict(self, task_index):
        temp_state_dict = {}
        for name in self.names_to_aggregate:
            trace = name.split('.')
            aggr_param = sum(
                [co_ef * self.get_params(trace, sub_model) for co_ef, sub_model in
                 zip(self.co_ef[task_index], self.models)])
            temp_state_dict[name] = aggr_param
        return temp_state_dict

    def self_update(self):
        self.update_co_ef()
        state_dicts = []
        for i in range(self.model_num):
            state_dicts.append(self.aggregate_as_state_dict(i))
        # then, load params in fake_models back to models
        for i in range(self.model_num):
            self.models[i].load_state_dict(state_dicts[i], strict=False)

    def forward(self, input, is_sep_learn=False):
        if not is_sep_learn:
            self.update_co_ef()
        outputs = {}

        for task_index in range(len(self.models)):
            if not is_sep_learn:  # joint learn
                self.aggregate(task_index)  # aggregate the encoders to get the encoder of fake_models[task_index]

                rep = self.fake_models[task_index].encoder(input)
                rep = self.fake_models[task_index].final_conv(rep)
                rep = self.fake_models[task_index].final_conv_bn(rep)
            else:
                rep = self.models[task_index].encoder(input)
                rep = self.models[task_index].final_conv(rep)
                rep = self.models[task_index].final_conv_bn(rep)
            #  use the original decoder (do not aggregate the decoders)
            outputs[self.models[task_index].task] = self.models[task_index].decoder(rep)

        return outputs



class XceptionTaskonomyBase(nn.Module):
    """
    Xception optimized for the ImageNet dataset, as specified in
    https://arxiv.org/pdf/1610.02357.pdf
    """

    def __init__(self, task: str, index, tot_num, size=1):
        """ Constructor
        Args:
        """
        super(XceptionTaskonomyBase, self).__init__()
        assert type(task) == str
        sizes = [32, 64, 128, 256, 728, 728, 728, 728, 728, 728, 728, 728, 728]
        assert size == 1
        if size == 1:
            sizes = [32, 64, 128, 256, 728, 728, 728, 728, 728, 728, 728, 728, 728]
        elif size == .2:
            sizes = [16, 32, 64, 256, 320, 320, 320, 320, 320, 320, 320, 320, 320]
        elif size == .3:
            sizes = [32, 64, 128, 256, 728, 728, 728, 728, 728, 728, 728, 728, 728]
        elif size == .4:
            sizes = [32, 64, 128, 256, 728, 728, 728, 728, 728, 728, 728, 728, 728]
        elif size == .5:
            sizes = [24, 48, 96, 192, 512, 512, 512, 512, 512, 512, 512, 512, 512]
        elif size == .8:
            sizes = [32, 64, 128, 248, 648, 648, 648, 648, 648, 648, 648, 648, 648]
        elif size == 2:
            sizes = [32, 64, 128, 256, 728, 728, 728, 728, 728, 728, 728, 728, 728]
        elif size == 4:
            sizes = [64, 128, 256, 512, 1456, 1456, 1456, 1456, 1456, 1456, 1456, 1456, 1456]

        self.encoder = Encoder(sizes=sizes)
        pre_rep_size = sizes[-1]

        self.task = task

        self.final_conv = SeparableConv2d(pre_rep_size, 512, 3, 1, 1)
        self.final_conv_bn = nn.BatchNorm2d(512)

        if task == 'segment_semantic':
            output_channels = 18
        elif task == 'depth_zbuffer':
            output_channels = 1
        elif task == 'normal':
            output_channels = 3
        elif task == 'edge_occlusion':
            output_channels = 1
        elif task == 'keypoints2d':
            output_channels = 1
        elif task == 'edge_texture':
            output_channels = 1
        elif task == 'reshading':
            output_channels = 1
        elif task == 'rgb':
            output_channels = 3
        elif task == 'principal_curvature':
            output_channels = 2
        else:
            raise NotImplementedError('Unknown task', task)

        self.decoder = Decoder(output_channels)

        # ------- init weights --------
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        # -----------------------------

        self.W = torch.nn.Parameter(torch.tensor(np.eye(tot_num)[index]), requires_grad=True)
        self.parameterized_W = None

    def forward(self, input):
        rep = self.encoder(input)
        rep = self.final_conv(rep)
        rep = self.final_conv_bn(rep)
        output = self.decoder(rep)
        return output


class XceptionMultiTask(nn.Module):
    def __init__(self, tasks: list, **kwargs):
        super().__init__()
        sizes = [32, 64, 128, 256, 728, 728, 728, 728, 728, 728, 728, 728, 728]
        self.encoder = Encoder(sizes=sizes)
        pre_rep_size = sizes[-1]
        self.tasks = tasks
        self.final_conv = SeparableConv2d(pre_rep_size, 512, 3, 1, 1)
        self.final_conv_bn = nn.BatchNorm2d(512)
        self.decoders = nn.ModuleList()
        for task in tasks:
            if task == 'segment_semantic':
                # taskonomy
                if 'num_seg_cls' in kwargs:
                    output_channels = kwargs['num_seg_cls']
                else:
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
            self.decoders.append(Decoder(output_channels))

    def forward(self, input):
        outputs = {}
        rep = self.encoder(input)
        rep = self.final_conv(rep)
        rep = self.final_conv_bn(rep)
        for task_index in range(len(self.tasks)):
            outputs[self.tasks[task_index]] = self.decoders[task_index](rep)
        return outputs


class XceptionMultiTaskMultiCase(nn.Module):
    def __init__(self, tasks: list):
        super().__init__()
        sizes = [32, 64, 128, 256, 728, 728, 728, 728, 728, 728, 728, 728, 728]
        self.encoder = Encoder(sizes=sizes)
        pre_rep_size = sizes[-1]
        self.tasks = tasks
        self.final_conv = SeparableConv2d(pre_rep_size, 512, 3, 1, 1)
        self.final_conv_bn = nn.BatchNorm2d(512)
        self.decoders = nn.ModuleList()
        for task in tasks:
            temp_task = task.split('/')[-1]
            if temp_task == 'segment_semantic':
                # taskonomy
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
            self.decoders.append(Decoder(output_channels))

    def forward(self, input):
        outputs = {}
        for task_index in range(len(self.tasks)):
            rep = self.encoder(input[self.tasks[task_index]])
            rep = self.final_conv(rep)
            rep = self.final_conv_bn(rep)
            #  use the original decoder (do not aggregate the decoders)
            outputs[self.tasks[task_index]] = self.decoders[task_index](rep)
        return outputs


class W_network(nn.Module):
    def __init__(self, model_num, layer_num=1):
        super().__init__()
        self.model_num = model_num
        self.layer_num = layer_num
        self.W_1 = nn.ParameterList([nn.Parameter(torch.tensor(np.eye(self.model_num), dtype=torch.float), requires_grad=True) for _ in range(layer_num)])
        self.W_2 = nn.ParameterList([nn.Parameter(torch.tensor(np.eye(self.model_num), dtype=torch.float), requires_grad=True) for _ in range(layer_num)])

    def forward(self, param_x):
        x = param_x
        for i in range(self.layer_num):
            x = F.relu(torch.matmul(x, self.W_1[i])) - F.relu(torch.matmul(-x, self.W_2[i]))
        return x

    def reset(self):
        for i in range(self.layer_num):
            self.W_1[i].data = torch.tensor(np.eye(self.model_num), dtype=torch.float).cuda()
            self.W_2[i].data = torch.tensor(np.eye(self.model_num), dtype=torch.float).cuda()


class W_network_Residual(nn.Module):
    def __init__(self, model_num, layer_num=1):
        super().__init__()
        self.model_num = model_num
        self.layer_num = layer_num
        self.W_1 = nn.ParameterList([nn.Parameter(torch.tensor(np.zeros((self.model_num, self.model_num)),
                                                               dtype=torch.float), requires_grad=True) for _ in range(layer_num)])

    def forward(self, param_x):
        x = param_x
        for i in range(self.layer_num):
            x = F.tanh(torch.matmul(x, self.W_1[i]))
        return param_x + x  # residual learning

    def reset(self):
        for i in range(self.layer_num):
            self.W_1[i].data = torch.tensor(np.zeros((self.model_num, self.model_num)), dtype=torch.float).cuda()


class XceptionTaskonomyJointLearnBetter(nn.Module):
    def __init__(self, tasks: list, size=1, ozan=False):
        super().__init__()
        self.model_num = len(tasks)
        self.models = nn.ModuleList([XceptionTaskonomySep(tasks[i], size, ozan) for i in range(self.model_num)])
        self.fake_models = nn.ModuleList([XceptionTaskonomySep(tasks[i], size, ozan) for i in range(self.model_num)])
        self.n_workers = self.model_num

        self.names_to_pop = [name for name, para in self.named_parameters() if name.startswith('fake')]
        self.names_to_pop_buffer = [name for name, para in self.named_buffers() if name.startswith('fake')]

        self.names_to_aggregate = [name for name, para in self.models[0].named_parameters() if
                                   name.startswith('encoder') or name.startswith('final')]

        self.names_to_aggregate_buffer = [name for name, para in self.models[0].named_buffers() if
                                          name.startswith('encoder') or name.startswith('final')]

        # self.W_network = W_network(self.model_num, layer_num=3)
        self.W_network = W_network_Residual(self.model_num, layer_num=3)
        # self.W = self.W_network.parameters()

    def set_buffer(self):
        for name in self.names_to_pop:
            trace = name.split('.')
            self.pop_params(trace, self)
        for name in self.names_to_pop_buffer:
            trace = name.split('.')
            self.pop_buffers(trace, self)
        for name in self.names_to_aggregate_buffer:
            trace = name.split('.')
            for i in range(self.model_num):
                self.set_params(trace, self.fake_models[i], self.get_params(trace, self.models[i]))

    @staticmethod
    def get_params(trace, target_model):
        temp = target_model
        for _ in range(len(trace)):
            temp = temp.__getattr__(trace[_])
        return temp

    @staticmethod
    def set_params(trace, target_model, param):
        temp = target_model
        for _ in range(len(trace) - 1):
            temp = temp.__getattr__(trace[_])
        temp.__setattr__(trace[-1], param)

    @staticmethod
    def pop_params(trace, target_model):
        temp = target_model
        for _ in range(len(trace) - 1):
            temp = temp.__getattr__(trace[_])
        temp._parameters.pop(trace[-1])

    @staticmethod
    def pop_buffers(trace, target_model):
        temp = target_model
        for _ in range(len(trace) - 1):
            temp = temp.__getattr__(trace[_])
        temp._buffers.pop(trace[-1])

    def aggregate(self, set_back=False):
        # TODO: use the W network output
        for name in self.names_to_aggregate:
            trace = name.split('.')
            params = torch.cat([self.get_params(trace, sub_model).unsqueeze(-1) for sub_model in self.models], dim=-1)
            aggr_param = self.W_network(params.reshape(-1, self.model_num)).reshape_as(params)
            for i in range(self.model_num):
                if not set_back:
                    self.set_params(trace, self.fake_models[i], aggr_param[..., i])
                else:
                    self.get_params(trace, self.models[i]).data = aggr_param[..., i]

    def self_update(self):
        self.aggregate(set_back=True)
        self.W_network.reset()

    def forward(self, input, is_sep_learn=False):
        if not is_sep_learn:
            self.aggregate()
        outputs = {}

        for task_index in range(len(self.models)):
            if not is_sep_learn:  # joint learn
                rep = self.fake_models[task_index].encoder(input)
                rep = self.fake_models[task_index].final_conv(rep)
                rep = self.fake_models[task_index].final_conv_bn(rep)
            else:
                rep = self.models[task_index].encoder(input)
                rep = self.models[task_index].final_conv(rep)
                rep = self.models[task_index].final_conv_bn(rep)
            #  use the original decoder (do not aggregate the decoders)
            outputs[self.models[task_index].task] = self.models[task_index].decoder(rep)

        return outputs

def xception_taskonomy_sep(task: str, size=1, ozan=False):
    return XceptionTaskonomySep(task, size, ozan)


def xception_taskonomy_jointlearn(tasks: list, size=1, ozan=False):
    return XceptionTaskonomyJointLearn(tasks, size, ozan)

def xception_multitask(tasks: list, **kwargs):
    return XceptionMultiTask(tasks=tasks, **kwargs)

def xception_taskonomy_jointlearn_better(tasks: list, size=1, ozan=False):
    return XceptionTaskonomyJointLearnBetter(tasks, size, ozan)

def xception_multitask_multicase(tasks: list):
    return XceptionMultiTaskMultiCase(tasks=tasks)