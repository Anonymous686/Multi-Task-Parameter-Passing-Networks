import pickle
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from collections import defaultdict


def get_splits(data_dir):
    if data_dir.endswith('tiny'):
        train_model_whitelist = 'train_models_tiny.txt'
        val_model_whitelist = 'val_models_tiny.txt'
        test_model_whitelist = 'test_models_tiny.txt'
    elif data_dir.endswith('medium'):
        train_model_whitelist = 'train_models_medium.txt'
        val_model_whitelist = 'val_models_medium.txt'
        test_model_whitelist = 'test_models_medium.txt'
    else:
        train_model_whitelist = 'train_models.txt'
        val_model_whitelist = 'val_models.txt'
        test_model_whitelist = 'val_models.txt'

    return 'splits/' + train_model_whitelist, 'splits/' + val_model_whitelist, 'splits/' + test_model_whitelist


def get_n_params(model):
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn * s
        pp += nn
    return pp


class Saver(object):
    def __init__(self, save_name, tasks, save_dir='history'):
        self.save_name = save_name
        self.tasks = tasks
        self.results = {'average': []}
        self.val_results = {'average': []}
        self.co_efs = []
        self.cur_accumulated_products = np.eye(len(self.tasks)) * 1.0
        self.accumulated_products = []
        self.norms = []
        self.save_dir = save_dir
        if self.save_dir[-1] == '/':
            self.save_dir = self.save_dir[:-1]
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)
        if not os.path.exists(self.save_dir + '/' + self.save_name):
            os.mkdir(self.save_dir + '/' + self.save_name)
        print('Save to ', self.save_dir + '/' + self.save_name)

    def append_co_ef(self, co_ef_matrix):
        self.co_efs.append(co_ef_matrix)
        self.cur_accumulated_products = np.matmul(co_ef_matrix, self.cur_accumulated_products)
        self.accumulated_products.append(self.cur_accumulated_products)
        self.norms.append(np.linalg.norm(co_ef_matrix - np.eye(len(self.tasks)) * 1.0))

    def append_val(self, res: dict):
        tot, cnt = 0, 0
        for _ in res:
            if _ == 'Loss':
                continue
            if _ not in self.val_results:
                self.val_results[_] = [float(res[_])]
                tot += float(res[_])
                cnt += 1
            else:
                self.val_results[_].append(float(res[_]))
                tot += float(res[_])
                cnt += 1
        if cnt == len(self.tasks):
            self.val_results['average'].append(tot / cnt)
        else:
            raise RuntimeError()

    def append(self, res: dict):
        tot, cnt = 0, 0
        for _ in res:
            if _ == 'Loss':
                continue
            if _ not in self.results:
                self.results[_] = []  # omit the first step
                continue
            else:
                assert type(res[_]) == torch.Tensor
                temp = res[_].item()
                self.results[_].append(temp)
                tot += temp
                cnt += 1
        if cnt == 0:
            return
        elif cnt == len(self.tasks):
            self.results['average'].append(tot / cnt)
            return
        else:
            raise RuntimeError()

    def save(self):
        print('Saving to ', self.save_dir + '/' + self.save_name + '/' + self.save_name + '.pkl')
        pickle.dump(
            (self.save_name, self.tasks, self.results, self.val_results, self.co_efs, self.accumulated_products, self.norms),
            open(self.save_dir + '/' + self.save_name + '/' + self.save_name + '.pkl', 'wb'))
        print('Saved !')

    def load(self, name):
        self.save_name, self.tasks, self.results, self.val_results, self.co_efs, self.accumulated_products, self.norms = \
            pickle.load(open(self.save_dir + '/' + name + '/' + name + '.pkl', 'rb'))

    def plot(self):
        plt.figure(figsize=(20, 10), dpi=8)
        plt.title(self.save_name)
        plt.xlabel('Step')
        plt.ylabel('Loss')
        epoch_num = len(list(self.results.values())[0])
        for _ in self.results:
            assert epoch_num == len(self.results[_])
            plt.plot(np.arange(epoch_num), self.results[_], label=_)
        plt.legend(loc='best')
        if not os.path.exists('figures'):
            os.mkdir('figures')
        plt.savefig('figures/' + self.save_name + '.pdf')



def get_average_learning_rate(optimizer):
    try:
        return optimizer.learning_rate
    except:
        s = 0
        for param_group in optimizer.param_groups:
            s += param_group['lr']
        return s / len(optimizer.param_groups)


class DataPrefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            next_input, next_target = next(self.loader)
            with torch.cuda.stream(self.stream):
                self.next_input = next_input.cuda(non_blocking=True)
                self.next_target = {key: val.cuda(non_blocking=True) for (key, val) in next_target.items()}
        except StopIteration:
            return

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        target = self.next_target
        self.preload()
        return input, target


class DataPrefetcherMultiGPU():
    def __init__(self, loader, devices, tasks):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.devices = devices
        self.preload()

    def preload(self):
        try:
            next_input, next_target = next(self.loader)
            with torch.cuda.stream(self.stream):
                self.next_input = [next_input.cuda(device=torch.device(_), non_blocking=True) for _ in self.devices]
                self.next_target = {key: val.cuda(device=torch.device(self.devices[0]), non_blocking=True) for (key, val) in next_target.items()}
        except StopIteration:
            return

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        target = self.next_target
        self.preload()
        return input, target


class DataPrefetcherMultiGPUMultiCase():
    def __init__(self, loader, devices):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.devices = devices
        self.preload()

    def preload(self):
        try:
            next_input, next_target = next(self.loader)
            with torch.cuda.stream(self.stream):
                cnt = 0
                self.next_input = {}
                for key, val in next_input.items():
                    self.next_input[key] = val.cuda(device=torch.device(self.devices[cnt]), non_blocking=True)
                    cnt += 1
                # self.next_input = [next_input.cuda(device=torch.device(_), non_blocking=True) for _ in self.devices]
                self.next_target = {key: val.cuda(device=torch.device(self.devices[0]), non_blocking=True) for (key, val) in next_target.items()}
        except StopIteration:
            return

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        target = self.next_target
        self.preload()
        return input, target


class DataPrefetcherMultiCase():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            next_input, next_target = next(self.loader)
            with torch.cuda.stream(self.stream):
                cnt = 0
                self.next_input = {}
                for key, val in next_input.items():
                    self.next_input[key] = val.cuda(non_blocking=True)
                    cnt += 1
                # self.next_input = [next_input.cuda(device=torch.device(_), non_blocking=True) for _ in self.devices]
                self.next_target = {key: val.cuda(non_blocking=True) for (key, val) in next_target.items()}
        except StopIteration:
            return

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        target = self.next_target
        self.preload()
        return input, target


class color:
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


def print_table(table_list, go_back=True):
    if len(table_list) == 0:
        print()
        print()
        return
    if go_back:
        print("\033[F", end='')
        print("\033[K", end='')
        for i in range(len(table_list)):
            print("\033[F", end='')
            print("\033[K", end='')

    lens = defaultdict(int)
    for i in table_list:
        for ii, to_print in enumerate(i):
            for title, val in to_print.items():
                lens[(title, ii)] = max(lens[(title, ii)], max(len(title), len(val)))

    for ii, to_print in enumerate(table_list[0]):
        for title, val in to_print.items():
            print('{0:^{1}}'.format(title, lens[(title, ii)]), end=" ")
    for i in table_list:
        print()
        for ii, to_print in enumerate(i):
            for title, val in to_print.items():
                print('{0:^{1}}'.format(val, lens[(title, ii)]), end=" ", flush=True)
    print()


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.std = 0
        self.sum = 0
        self.sumsq = 0
        self.count = 0
        self.lst = []

    def update(self, val, n=1):
        self.val = float(val)
        self.sum += float(val) * n
        self.count += n
        self.avg = self.sum / self.count
        self.lst.append(self.val)
        self.std = np.std(self.lst)



class DataPrefetcherMultiGPU_IMG():
    def __init__(self, loader, devices, tasks):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.devices = devices
        self.preload()

    def preload(self):
        try:
            next_image, next_input, next_target = next(self.loader)
            with torch.cuda.stream(self.stream):
                self.next_image = next_image
                self.next_input = [next_input.cuda(device=torch.device(_), non_blocking=True) for _ in self.devices]
                self.next_target = {key: val.cuda(device=torch.device(self.devices[0]), non_blocking=True) for (key, val) in next_target.items()}
        except StopIteration:
            return

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        target = self.next_target
        image = self.next_image
        self.preload()
        return image, input, target


class DataPrefetcher_IMG():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            next_image, next_input, next_target = next(self.loader)
            with torch.cuda.stream(self.stream):
                self.next_image = next_image
                self.next_input = next_input.cuda(non_blocking=True)
                self.next_target = {key: val.cuda(non_blocking=True) for (key, val) in next_target.items()}
        except StopIteration:
            return

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        image = self.next_image
        input = self.next_input
        target = self.next_target
        self.preload()
        return image, input, target