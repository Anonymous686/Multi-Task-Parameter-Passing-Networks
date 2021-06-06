import torch.nn as nn
import torch.nn.functional as F
import torch
from .xception_taskonomy_sep import XceptionTaskonomySep
from .resnet_taskonomy_sep import ResNetSep

__all__ = ['ppnet_distributed_nonlinear']


class W_network_NonLinear(nn.Module):
    def __init__(self, model_num, non_linearity=True, layer_num=1, channel_wise=0, layer_shape=None, residual=False,
                 col_normalize=False):
        super().__init__()
        assert layer_shape is None or len(layer_shape) == 1 or len(layer_shape) == 4
        self.residual = residual
        self.model_num = model_num
        self.layer_num = layer_num
        self.channel_wise = channel_wise
        self.hidden_dim = 64
        self.non_linearity = non_linearity
        self.col_normalize = col_normalize
        # here
        # self.non_linearity = False
        # non_linearity = False
        # self.col_normalize = True
        # col_normalize = True

        if non_linearity:
            assert residual
        if col_normalize:
            assert not residual

        self.channels = None
        if channel_wise == 0:
            if self.residual:
                self.W_base = nn.ParameterList(
                    [nn.Parameter(torch.zeros((self.model_num, self.model_num), dtype=torch.float), requires_grad=True)
                     for _
                     in
                     range(layer_num)])
            else:
                self.W_base = nn.ParameterList(
                    [nn.Parameter(torch.eye(self.model_num, dtype=torch.float), requires_grad=True)
                     for _
                     in
                     range(layer_num)])
        else:
            assert layer_shape is not None
            if channel_wise == 1:
                self.channels = int(layer_shape[0])
            elif channel_wise == 2:
                if len(layer_shape) < 2:
                    raise RuntimeError('BN layers weight shape len = 1, '
                                       'cannot use channel_wise=2 option together with conv_only=False')
                self.channels = int(layer_shape[0]) * int(layer_shape[1])
            else:
                raise NotImplementedError()
            W_base = []
            # Special case when there's only one layer
            if layer_num == 1:
                self.hidden_dim = self.model_num * self.channels

            for i in range(layer_num):
                if i == 0:
                    if self.residual:
                        W_base.append(
                            nn.Parameter(torch.zeros((self.model_num * self.channels, self.hidden_dim), dtype=torch.float),
                                         requires_grad=True))
                    else:
                        # raise NotImplementedError()
                        assert self.model_num * self.channels == self.hidden_dim
                        W_base.append(
                            nn.Parameter(torch.eye(self.model_num * self.channels, dtype=torch.float),
                                         requires_grad=True))
                elif i < layer_num - 1:
                    if self.residual:
                        W_base.append(
                            nn.Parameter(torch.rand((self.hidden_dim, self.hidden_dim), dtype=torch.float),
                                         requires_grad=True))
                    else:
                        raise NotImplementedError()
                else:
                    if self.residual:
                        W_base.append(
                            nn.Parameter(torch.rand((self.hidden_dim, self.model_num * self.channels), dtype=torch.float),
                                         requires_grad=True))
                    else:
                        raise NotImplementedError()

            self.W_base = nn.ParameterList(W_base)

        # self.alpha = nn.Parameter(torch.zeros(1), requires_grad=True)

    def forward(self, param_x):
        # transform param_x to feed into W networks
        if len(param_x.shape) == 2:
            layer_type = 1
            b, N = int(param_x.shape[0]), int(param_x.shape[1])
        elif len(param_x.shape) == 5:
            layer_type = 2
            out_channels, in_channels, w, h, N = int(param_x.shape[0]), int(param_x.shape[1]), int(param_x.shape[2]), \
                                                 int(param_x.shape[3]), int(param_x.shape[4])
        else:
            raise RuntimeError()

        if self.channel_wise == 0:
            pass
        elif self.channel_wise == 1:
            if len(param_x.shape) == 2:
                param_x = param_x.reshape(1, -1)
            elif len(param_x.shape) == 5:
                param_x = param_x.permute(1, 2, 3, 0, 4).reshape(-1, out_channels * N)
            else:
                raise RuntimeError()
        else:
            # must be conv layers
            assert len(param_x.shape) == 5
            param_x = param_x.permute(2, 3, 0, 1, 4).reshape(-1, out_channels * in_channels * N)

        x = param_x  # param_x: either (b, N) or (c_out, c_in, w, h, N)
        # x = self.alpha * x
        for i in range(self.layer_num):
            if self.col_normalize:
                if self.channel_wise == 0:
                    temp_W = self.W_base[i] / torch.sum(self.W_base[i], dim=0, keepdim=True).expand_as(self.W_base[i])
                elif self.channel_wise == 1:
                    temp_W = self.W_base[i].reshape(self.model_num, self.channels, self.model_num, self.channels)\
                        .permute(0, 2, 1, 3).reshape(self.model_num, -1)
                    temp_W = temp_W / torch.sum(temp_W, dim=0, keepdim=True).expand_as(temp_W)
                    temp_W = temp_W.reshape(self.model_num, self.model_num, self.channels, self.channels)\
                        .permute(0, 2, 1, 3).reshape(self.model_num * self.channels, -1)
                else:
                    raise NotImplementedError()
                x = torch.matmul(x, temp_W)
            else:
                x = torch.matmul(x, self.W_base[i])

            if self.non_linearity:
                x = torch.tanh(x)

        if self.residual:
            ret = param_x + x
        else:
            ret = x

        # transform ret back to the original shape
        if self.channel_wise == 0:
            pass
        elif self.channel_wise == 1:
            if layer_type == 1:
                ret = ret.reshape(b, N)
            elif layer_type == 2:
                ret = ret.reshape(in_channels, w, h, out_channels, N).permute(3, 0, 1, 2, 4)
            else:
                raise RuntimeError()
        else:
            # must be conv layers
            assert layer_type == 2
            ret = ret.reshape(w, h, out_channels, in_channels, N).permute(2, 3, 0, 1, 4)

        return ret

    def reset(self):
        # find the most efficient way
        assert len(self.W_base) == 1
        for i in range(len(self.W_base)):
            if self.residual:
                self.W_base[i].data = self.W_base[i].data.zero_()
            else:
                if self.channel_wise == 0:
                    self.W_base[i].data = torch.eye(self.model_num, dtype=torch.float).to(self.W_base[i].data)
                else:
                    self.W_base[i].data = torch.eye(self.model_num * self.channels, dtype=torch.float).to(self.W_base[i].data)


class PPNetDistributedNonLinear(torch.nn.Module):
    def __init__(self, tasks: list, size=1, devices=None, conv_only=False, layer_wise=False, channel_wise=0,
                 residual=False, backbone='xception'):
        super(PPNetDistributedNonLinear, self).__init__()
        self.model_num = len(tasks)
        self.tasks = tasks
        self.backbone = backbone
        self.conv_only, self.layer_wise = conv_only, layer_wise
        self.channel_wise = channel_wise
        self.residual = residual

        if devices is None:
            print('Distributed model reduced to single gpu training')
            self.devices = ['cuda:0' for _ in range(self.model_num)]
        else:
            self.devices = devices
            print('Distributed model training with configuration: ', self.devices)

        if self.backbone == 'xception':
            self.models = nn.ModuleList([XceptionTaskonomySep(tasks[i], size=size)
                                         for i in range(self.model_num)])
            self.fake_models = nn.ModuleList([XceptionTaskonomySep(tasks[i], size=size)
                                              for i in range(self.model_num)])
        elif self.backbone == 'resnet':
            # currently, using resnet34
            self.models = nn.ModuleList([ResNetSep(tasks[i], size=size)
                                         for i in range(self.model_num)])
            self.fake_models = nn.ModuleList([ResNetSep(tasks[i], size=size)
                                              for i in range(self.model_num)])
        else:
            raise NotImplementedError('Unknown backbone:', self.backbone)


        self.names_to_pop = [name for name, para in self.named_parameters() if name.startswith('fake')]
        self.names_to_pop_buffer = [name for name, para in self.named_buffers() if name.startswith('fake')]

        if not self.conv_only:
            self.names_to_aggregate = [name for name, para in self.models[0].named_parameters() if
                                       name.startswith('encoder') or name.startswith('final')]
            self.names_to_reuse = []
        else:
            self.names_to_aggregate = [name for name, para in self.models[0].named_parameters() if
                                       (name.startswith('encoder') or name.startswith('final')) and len(para.shape) == 4]
            self.names_to_reuse = [name for name, para in self.models[0].named_parameters() if
                                       (name.startswith('encoder') or name.startswith('final')) and len(para.shape) < 4]
            # print('params to reuse', self.names_to_reuse)

        self.names_to_aggregate_buffer = [name for name, para in self.models[0].named_buffers() if
                                          name.startswith('encoder') or name.startswith('final')]
        # print(self.names_to_aggregate_buffer)
        # print(self.names_to_aggregate)

        for name in self.names_to_pop:
            trace = name.split('.')
            self.pop_params(trace, self)

        for name in self.names_to_pop_buffer:
            trace = name.split('.')
            self.pop_buffers(trace, self)

        # construct W networks
        print('Constrcting W networks')

        W_networks = {}
        if self.layer_wise:
            print('Layer-wise W Network')
            print('channel_wise set to', self.channel_wise)
            for name in self.names_to_aggregate:
                param_shape = self.get_params(name.split('.'), self.models[0]).shape
                if self.conv_only:
                    assert len(param_shape) == 4
                cur_W_network = W_network_NonLinear(model_num=self.model_num,
                                                    layer_num=1, channel_wise=self.channel_wise,
                                                    layer_shape=param_shape,
                                                    residual=residual)
                W_networks[name.replace('.', '_')] = cur_W_network
        else:
            print('Shared W Network')
            # channel wise must be zero because shapes are different in different layers
            print('channel_wise set to 0 because layer_wise is False')
            cur_W_network = W_network_NonLinear(model_num=self.model_num,
                                                layer_num=1, channel_wise=0, layer_shape=None, residual=residual)
            W_networks['shared'] = cur_W_network
        self.W_networks = nn.ModuleDict(W_networks)
        # count total parameter number in W networks
        cnt = 0
        for name in self.W_networks:
            for param in self.W_networks[name].parameters():
                cnt += param.numel()
        print('Parameters in W networks: ', cnt)
        # for name, param in self.named_parameters():
        #     if name.startswith('W'):
        #         print(name)

    def to_cuda(self):
        for i in range(self.model_num):
            self.models[i] = self.models[i].to(self.devices[i])
            self.fake_models[i] = self.fake_models[i].to(self.devices[i])
        self.W_networks = self.W_networks.to(self.devices[0])
        self.set_params_and_buffers()

    def set_params_and_buffers(self):
        print('setting reused params and buffers')
        for name in self.names_to_aggregate_buffer:
            trace = name.split('.')
            for i in range(self.model_num):
                self.set_params(trace, self.fake_models[i], self.get_params(trace, self.models[i]))
        for name in self.names_to_reuse:
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
        for name in self.names_to_aggregate:
            trace = name.split('.')
            params = torch.cat([self.get_params(trace, sub_model).to(self.devices[0]).unsqueeze(-1)
                                for sub_model in self.models], dim=-1)  # shape: out_c, in_c, w, h, N
            if self.layer_wise:
                aggr_param = self.W_networks[name.replace('.', '_')](params)
            else:
                aggr_param = self.W_networks['shared'](params)
            assert aggr_param.shape == params.shape
            for i in range(len(self.models)):
                if not set_back:
                    self.set_params(trace, self.fake_models[i], aggr_param[..., i].to(self.devices[i]))
                else:
                    self.get_params(trace, self.models[i]).data = aggr_param[..., i].to(self.devices[i])

    def self_update(self):
        self.aggregate(set_back=True)
        for name in self.W_networks:
            self.W_networks[name].reset()

    def forward(self, input, is_sep_learn=False):
        outputs = {}
        if not is_sep_learn:
            self.aggregate()

        for task_index in range(len(self.models)):
            if not is_sep_learn:  # joint learn
                rep = self.fake_models[task_index].encoder(input[task_index])
                # rep = self.fake_models[task_index].encoder(input[self.tasks[task_index]])
                if self.backbone == 'xception':
                    rep = self.fake_models[task_index].final_conv(rep)
                    rep = self.fake_models[task_index].final_conv_bn(rep)
            else:
                rep = self.models[task_index].encoder(input[task_index])
                # rep = self.models[task_index].encoder(input[self.tasks[task_index]])
                if self.backbone == 'xception':
                    rep = self.models[task_index].final_conv(rep)
                    rep = self.models[task_index].final_conv_bn(rep)
            #  use the original decoder (do not aggregate the decoders)
            outputs[self.models[task_index].task] = self.models[task_index].decoder(rep).to(self.devices[0])

        return outputs


def ppnet_distributed_nonlinear(tasks: list, size=1, devices=None, **kwargs):
    return PPNetDistributedNonLinear(tasks=tasks, size=size, devices=devices, **kwargs)