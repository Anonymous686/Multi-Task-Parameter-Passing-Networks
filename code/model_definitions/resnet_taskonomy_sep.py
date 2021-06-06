import torch.nn as nn
import math
import torch
import numpy as np

__all__ = ['resnet18_taskonomy_sep', 'resnet_taskonomy_jointlearn', 'resnet_multitask', 'ResNetSep', 'resnet_multitask_multicase',
           'ResNetSepSimple']


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # expansion = 4
    # modify here
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNetEncoder(nn.Module):

    def __init__(self, block, layers, widths=None, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNetEncoder, self).__init__()
        if widths is None:
            widths = [64, 128, 256, 512]
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, widths[0], layers[0])
        self.layer2 = self._make_layer(block, widths[1], layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, widths[2], layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, widths[3], layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x


class Decoder(nn.Module):
    def __init__(self, output_channels=32, base_match=512):
        super(Decoder, self).__init__()

        self.output_channels = output_channels

        self.relu = nn.ReLU(inplace=True)

        self.upconv0 = nn.ConvTranspose2d(base_match, 256, 2, 2)
        self.bn_upconv0 = nn.BatchNorm2d(256)
        self.conv_decode0 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn_decode0 = nn.BatchNorm2d(256)
        self.upconv1 = nn.ConvTranspose2d(256, 128, 2, 2)
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
        self.upconv4 = nn.ConvTranspose2d(48, 32, 2, 2)
        self.bn_upconv4 = nn.BatchNorm2d(32)
        self.conv_decode4 = nn.Conv2d(32, output_channels, 3, padding=1)

    def forward(self, representation):
        x = self.upconv0(representation)
        x = self.bn_upconv0(x)
        x = self.relu(x)
        x = self.conv_decode0(x)
        x = self.bn_decode0(x)
        x = self.relu(x)
        x = self.upconv1(x)
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
        x = self.upconv4(x)
        x = self.bn_upconv4(x)
        x = self.relu(x)
        x = self.conv_decode4(x)

        return x


class ResNetSep(nn.Module):
    def __init__(self, task, size=1, **kwargs):
        super(ResNetSep, self).__init__()
        block = BasicBlock
        # Bottleneck for resnet50
        # block = Bottleneck
        # Now, resnet18 [2, 2, 2, 2]
        # here, resnet34 [3, 4, 6, 3]
        layers = [3, 4, 6, 3]
        if size == 1:
            self.encoder = ResNetEncoder(block, layers, **kwargs)
        elif size == 2:
            self.encoder = ResNetEncoder(block, layers, [96, 192, 384, 720], **kwargs)
        elif size == 3:
            self.encoder = ResNetEncoder(block, layers, [112, 224, 448, 880], **kwargs)
        elif size == 0.5:
            self.encoder = ResNetEncoder(block, layers, [48, 96, 192, 360], **kwargs)
        self.task = task
        temp_task = task.split('/')[-1]
        if temp_task == 'segment_semantic':
            # taskonomy
            output_channels = 18
            # nyu_v2
            # output_channels = 40
            # cityscapes
            # output_channels = 19
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

        if size == 1:
            decoder = Decoder(output_channels)
        elif size == 2:
            decoder = Decoder(output_channels, base_match=720)
        elif size == 3:
            decoder = Decoder(output_channels, base_match=880)
        elif size == 0.5:
            decoder = Decoder(output_channels, base_match=360)
        else:
            raise NotImplementedError('Unknown size for ResNet:', size)

        self.decoder = decoder

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
        output = self.decoder(rep)
        return output



class ResNetSepSimple(nn.Module):
    def __init__(self, task, size=1, num_seg_cls=19, **kwargs):
        super(ResNetSepSimple, self).__init__()
        block = BasicBlock
        # Bottleneck for resnet50
        # block = Bottleneck
        # Now, resnet18 [2, 2, 2, 2]
        # here, resnet34 [3, 4, 6, 3]
        layers = [3, 4, 6, 3]
        if size == 1:
            self.encoder = ResNetEncoder(block, layers, **kwargs)
        elif size == 2:
            self.encoder = ResNetEncoder(block, layers, [96, 192, 384, 720], **kwargs)
        elif size == 3:
            self.encoder = ResNetEncoder(block, layers, [112, 224, 448, 880], **kwargs)
        elif size == 0.5:
            self.encoder = ResNetEncoder(block, layers, [48, 96, 192, 360], **kwargs)
        self.task = task
        temp_task = task.split('/')[-1]
        if temp_task == 'segment_semantic':
            output_channels = num_seg_cls
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

        if size == 1:
            decoder = Decoder(output_channels)
        elif size == 2:
            decoder = Decoder(output_channels, base_match=720)
        elif size == 3:
            decoder = Decoder(output_channels, base_match=880)
        elif size == 0.5:
            decoder = Decoder(output_channels, base_match=360)
        else:
            raise NotImplementedError('Unknown size for ResNet:', size)

        self.decoder = decoder

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
        output = self.decoder(rep)
        return output


def resnet18_taskonomy_sep(task):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return ResNetSep(layers=[2, 2, 2, 2], task=task)



class ResNetTaskonomyJointLearn(nn.Module):
    def __init__(self, tasks: list, size=1):
        super().__init__()
        self.model_num = len(tasks)
        self.models = nn.ModuleList([ResNetSep(task=tasks[i], size=size) for i in range(self.model_num)])
        self.fake_models = nn.ModuleList([ResNetSep(task=tasks[i], size=size) for i in range(self.model_num)])
        self.n_workers = self.model_num
        # init W
        # W starts from uniform distribution
        # self.W = torch.nn.Parameter(torch.tensor(np.ones((self.n_workers, self.n_workers)) * 1.0), requires_grad=True)
        # W starts from self-loop
        self.W = torch.nn.Parameter(torch.tensor(np.eye(self.n_workers) * 1.0), requires_grad=True)

        self.co_ef = None

        self.names_to_pop = [name for name, para in self.named_parameters() if name.startswith('fake')]
        self.names_to_pop_buffer = [name for name, para in self.named_buffers() if name.startswith('fake')]

        self.names_to_aggregate = [name for name, para in self.models[0].named_parameters() if name.startswith('encoder')]
        self.names_to_aggregate_buffer = [name for name, para in self.models[0].named_buffers() if name.startswith('encoder')]

    def set_buffer(self):
        print('setting buffer')
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

    def update_co_ef(self):
        self.co_ef = self.get_co_ef()

    def get_co_ef(self):
        return self.W / self.W.sum(dim=-1, keepdim=True).expand_as(self.W)

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
            else:
                rep = self.models[task_index].encoder(input)
            #  use the original decoder (do not aggregate the decoders)
            outputs[self.models[task_index].task] = self.models[task_index].decoder(rep)

        return outputs


class ResNetMultiTask(nn.Module):
    def __init__(self, tasks: list, **kwargs):
        super().__init__()
        block = BasicBlock
        # layers = [2, 2, 2, 2]
        layers = [3, 4, 6, 3]
        self.encoder = ResNetEncoder(block, layers)
        self.decoders = nn.ModuleList()
        self.tasks = tasks
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
        for task_index in range(len(self.tasks)):
            outputs[self.tasks[task_index]] = self.decoders[task_index](rep)
        return outputs



class ResNetMultiTaskMultiCase(nn.Module):
    def __init__(self, tasks: list):
        super().__init__()
        block = BasicBlock
        # layers = [2, 2, 2, 2]
        layers = [3, 4, 6, 3]
        self.encoder = ResNetEncoder(block, layers)
        self.decoders = nn.ModuleList()
        self.tasks = tasks
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
            #  use the original decoder (do not aggregate the decoders)
            outputs[self.tasks[task_index]] = self.decoders[task_index](rep)

        return outputs


def resnet_taskonomy_jointlearn(tasks: list, size=1):
    return ResNetTaskonomyJointLearn(tasks=tasks, size=size)


def resnet_multitask(tasks: list, **kwargs):
    return ResNetMultiTask(tasks=tasks, **kwargs)


def resnet_multitask_multicase(tasks: list):
    return ResNetMultiTaskMultiCase(tasks=tasks)