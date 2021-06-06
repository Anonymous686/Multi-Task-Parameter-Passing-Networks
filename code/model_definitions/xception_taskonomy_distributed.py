import math
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import torch
import time
import numpy as np
from .xception_taskonomy_sep import XceptionTaskonomyBase

__all__ = ['xception_taskonomy_jointlearn_distributed', ]


class XceptionTaskonomyJointLearnDistributed(torch.nn.Module):
    def __init__(self, tasks: list, size=1, devices=None):
        super(XceptionTaskonomyJointLearnDistributed, self).__init__()
        self.model_num = len(tasks)
        self.tasks = tasks
        if devices is None:
            print('Distributed model reduced to single gpu training')
            self.devices = ['cuda:0' for _ in range(self.model_num)]
        else:
            self.devices = devices
            print('Distributed model training with configuration: ', self.devices)

        self.models = nn.ModuleList([XceptionTaskonomyBase(tasks[i], index=i, tot_num=self.model_num, size=size)
                                    .to(self.devices[i]) for i in range(self.model_num)])
        self.fake_models = nn.ModuleList([XceptionTaskonomyBase(tasks[i], index=i, tot_num=self.model_num, size=size)
                                         .to(self.devices[i]) for i in range(self.model_num)])


        self.names_to_pop = [name for name, para in self.named_parameters() if name.startswith('fake')]
        self.names_to_pop_buffer = [name for name, para in self.named_buffers() if name.startswith('fake')]

        self.names_to_aggregate = [name for name, para in self.models[0].named_parameters() if
                                   name.startswith('encoder') or name.startswith('final')]
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

        self.Ws = [name for name, param in self.named_parameters() if 'W' in name.split('.')]
        print(self.Ws)
        self.Ws_params = [param for name, param in self.named_parameters() if name in self.Ws]
        self.set_buffer()

    def set_buffer(self):
        print('setting buffer')
        for name in self.names_to_aggregate_buffer:
            trace = name.split('.')
            for i in range(self.model_num):
                self.set_params(trace, self.fake_models[i], self.get_params(trace, self.models[i]))

    def update_parameterized_Ws(self):
        # tot_W = torch.stack([self.models[_].W.to(self.devices[0]) for _ in range(self.model_num)], dim=0)
        # sym_W = (tot_W + tot_W.T) / 2
        # D = torch.diag(1 / torch.sqrt(torch.sum(sym_W, dim=1)))
        # W = D @ sym_W @ D
        # for i in range(self.model_num):
        #     self.models[i].parameterized_W = W[i].to(self.devices[i])

        for i in range(self.model_num):
            temp_W = self.models[i].W
            self.models[i].parameterized_W = temp_W / temp_W.sum(dim=-1, keepdim=True).expand_as(temp_W)

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
                [co_ef * self.get_params(trace, sub_model).to(self.devices[task_index]) for co_ef, sub_model in
                 zip(self.models[task_index].parameterized_W, self.models)])
            self.set_params(trace, self.fake_models[task_index], aggr_param)

    def aggregate_as_state_dict(self, task_index):
        temp_state_dict = {}
        for name in self.names_to_aggregate:
            trace = name.split('.')
            aggr_param = sum(
                [co_ef * self.get_params(trace, sub_model).to(self.devices[task_index]) for co_ef, sub_model in
                 zip(self.models[task_index].parameterized_W, self.models)])
            temp_state_dict[name] = aggr_param
        return temp_state_dict

    def self_update(self):
        self.update_parameterized_Ws()
        state_dicts = []
        for i in range(self.model_num):
            state_dicts.append(self.aggregate_as_state_dict(i))
        # then, load params in fake_models back to models
        for i in range(self.model_num):
            self.models[i].load_state_dict(state_dicts[i], strict=False)

    def forward(self, input, is_sep_learn=False):
        if not is_sep_learn:
            self.update_parameterized_Ws()
        outputs = {}
        if not is_sep_learn:
            for task_index in range(len(self.models)):
                self.aggregate(task_index)  # aggregate the encoders to get the encoder of fake_models[task_index]

        for task_index in range(len(self.models)):
            if not is_sep_learn:  # joint learn
                rep = self.fake_models[task_index].encoder(input[task_index])
                rep = self.fake_models[task_index].final_conv(rep)
                rep = self.fake_models[task_index].final_conv_bn(rep)
            else:
                rep = self.models[task_index].encoder(input[task_index])
                rep = self.models[task_index].final_conv(rep)
                rep = self.models[task_index].final_conv_bn(rep)
            #  use the original decoder (do not aggregate the decoders)
            outputs[self.models[task_index].task] = self.models[task_index].decoder(rep).to(self.devices[0])

        return outputs


def xception_taskonomy_jointlearn_distributed(tasks: list, size=1, devices=None):
    return XceptionTaskonomyJointLearnDistributed(tasks, size, devices)
