import argparse
import time
import platform
import torch.backends.cudnn as cudnn
import model_definitions as models
from utils import *

from baseline_loader import NYU_v2, CityScapes
from baseline_losses import get_baseline_losses_and_tasks

import torch.nn.functional as F
from sklearn.metrics import confusion_matrix

import warnings

warnings.filterwarnings('ignore')


model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch Taskonomy Training')
parser.add_argument('--data_dir', '-d', dest='data_dir', required=True,
                    help='path to training set')
parser.add_argument('--arch', '-a', metavar='ARCH', required=True,
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (required)')
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    help='mini-batch size (default: 64)')
parser.add_argument('--tasks', '-ts', default='sdnkt', dest='tasks',
                    help='which tasks to train on')
parser.add_argument('--model_dir', default='saved_models', dest='model_dir',
                    help='where to save models')
parser.add_argument('--image-size', default=256, type=int,
                    help='size of image side (images are square)')
parser.add_argument('-j', '--workers', default=4, type=int,
                    help='number of data loading workers (default: 4)')
parser.add_argument('-pf', '--print_frequency', default=1, type=int,
                    help='how often to print output')
parser.add_argument('--epochs', default=100, type=int,
                    help='maximum number of epochs to run')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '-wd', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('-n', '--experiment_name', default='', type=str,
                    help='name to prepend to experiment saves.')
parser.add_argument('-v', '--validate', dest='validate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('-t', '--test', dest='test', action='store_true',
                    help='evaluate model on test set')
parser.add_argument('--pretrained', dest='pretrained', default='',
                    help='use pre-trained model')
parser.add_argument('-vb', '--virtual-batch-multiplier', default=1, type=int,
                    metavar='N', help='number of forward/backward passes per parameter update')
parser.add_argument('-na', '--no_augment', action='store_true',
                    help='Run model fp16 mode.')
parser.add_argument('-s', '--seed', default=87, type=int,
                    help='Fix random seed.')
parser.add_argument('-save', '--save', action='store_true',
                    help='save model weights')
parser.add_argument('-sd', '--save_dir', default='history', type=str,
                    help='The save directory for statistics and ckpts')

parser.add_argument('-st', '--self_training_count', default=1000000, type=int,
                    help='The number of self-training-steps/epochs performed.')
parser.add_argument('-ct', '--co_training_count', default=0, type=int,
                    help='The number of co-training-steps/epochs performed.')
parser.add_argument('--epoch_wise', action='store_true',
                    help='should self/co training be epoch-wise or step-wise, step-wise if not specified')
parser.add_argument('--no_em', action='store_true',
                    help='should co-training only for W network or for all (E+D+W), only for W network if not specified')

parser.add_argument('-lr', '--learning-rate', dest='lr', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('-wlr', '--W-learning-rate', default=0.1, type=float, help='W learning rate')

parser.add_argument('--devices', default=None, type=str,
                    help='The devices for the tasks to conduct distributed training')

parser.add_argument('--conv_only', action='store_true',
                    help='should aggregation only on conv layers')
parser.add_argument('--layer_wise', action='store_true',
                    help='should W network be layer-wise')
parser.add_argument('--channel_wise', action='store_true',
                    help='should W network be channel-wise')
parser.add_argument('--residual', action='store_true',
                    help='should W network be residual')
parser.add_argument('--backbone', type=str,
                    help='The backbone model, currently supporting xception and resnet34')

parser.add_argument('--test_weight', type=str, default=None, help='The weight to test')

parser.add_argument('--test_interval', default=10, type=int,
                    help='Testing interval for metrics like mIoU.')
parser.add_argument('--opt_type', default='SGD', type=str,
                    help='Optimizer type.')
parser.add_argument('--num_seg_cls', default=19, type=int,
                    help='Semantic segmentation class')

parser.add_argument('--decay_interval', default=50, type=int, help='Interval for lr decay of self-training')
parser.add_argument('--gamma', default=1.0, type=float, help='the decay gamma of the lr of self-training')

parser.add_argument('--loss_weight', type=str, default=None, help='The loss weight of tasks.')


def main(args):
    """
    The main function handling the overall training / validation / testing logic.
    :param args: hyper-parameters and arguments
    """
    print(args)
    print('starting on', platform.node())
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        print('cuda gpus:', os.environ['CUDA_VISIBLE_DEVICES'])

    # fix random seed
    # cudnn.benchmark = False
    # cudnn.deterministic = True
    # np.random.seed(args.seed)
    # torch.manual_seed(args.seed)
    # torch.cuda.manual_seed_all(args.seed)
    assert len(args.tasks) == len(args.loss_weight.split(','))
    loss_weights = args.loss_weight.split(',')
    loss_weights = {args.tasks[_]: int(loss_weights[_]) for _ in range(len(args.tasks))}
    args.loss_weight = loss_weights

    taskonomy_loss, losses, criteria, taskonomy_tasks = get_baseline_losses_and_tasks(args)

    print("including the following tasks:", list(losses.keys()))
    criteria['Loss'] = taskonomy_loss
    print('data_dir =', args.data_dir)
    # Initialize a saver for saving results (training curves, etc.)
    saver = Saver(save_name=args.experiment_name, tasks=taskonomy_tasks, save_dir=args.save_dir)

    # load split
    if 'nyu_v2' in args.data_dir:
        train_dataset = NYU_v2(dataroot=args.data_dir, mode='train')
    elif 'cityscapes' in args.data_dir:
        train_dataset = CityScapes(dataroot=args.data_dir, mode='train')
        # train_dataset = CityScapes(dataroot=args.data_dir, mode='train', crop_h=321, crop_w=321)
    else:
        raise NotImplementedError()

    print('Found', len(train_dataset), 'training instances.')
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, sampler=None)

    val_loader = get_eval_loader(args.data_dir, args)

    devices = args.devices
    if devices is not None:
        devices = devices.split(',')
        for i in range(len(devices)):
            devices[i] = 'cuda:' + devices[i]

    print("=> creating model '{}'".format(args.arch))
    model = models.__dict__[args.arch](tasks=list(losses.keys()), devices=devices, conv_only=args.conv_only,
                                       layer_wise=args.layer_wise, channel_wise=1 if args.channel_wise else 0,
                                       residual=args.residual, backbone=args.backbone, num_seg_cls=args.num_seg_cls)
    # model.set_buffer()
    print("Model has", get_n_params(model), "parameters")

    # ckpt = 'epoch100_valloss0.49738261043423354.pth.tar'
    # print('Loading pretrained params from =>', ckpt)
    # # cur_path = 'history/sep/pretrained'
    # cur_path = 'history/sep_mid/pretrained'
    # checkpoint_file = torch.load(os.path.join(cur_path, ckpt), map_location='cpu')
    # checkpoint_file = {name: checkpoint_file[name] for name in checkpoint_file if not name.startswith('fake')}
    # model.load_state_dict(checkpoint_file, strict=False)
    print('porting to cuda gpus')
    # also set buffers and reused parameters here
    model.to_cuda()
    # Initialize the optimizers: optimizer for the params except W,  optimizer_W for W
    if args.opt_type == 'SGD':
        optimizer = torch.optim.SGD([param for name, param in model.named_parameters() if not name.startswith('W_networks')],
                                    lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.opt_type == 'Adam':
        optimizer = torch.optim.Adam([param for name, param in model.named_parameters() if not name.startswith('W_networks')],
                                     betas=(0.5, 0.999), lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise NotImplementedError()
    optimizer_W = torch.optim.SGD([param for name, param in model.named_parameters() if name.startswith('W_networks')],
                                  lr=args.W_learning_rate)
    # Initialize the scheduler for self-training optimizer
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.decay_interval, gamma=args.gamma)

    print('Virtual batch size =', args.batch_size * args.virtual_batch_multiplier)
    print('Training starts')
    trainer = Trainer(train_loader, val_loader, model, optimizer, optimizer_W, scheduler, criteria, saver, args)
    # trainer.validate([{}])

    if args.test_weight is None:
        trainer.train()
        print('Saving results')
        saver.save()
        with open(args.save_dir + '/' + args.experiment_name + '/metrics.pkl', 'wb') as f:
            pickle.dump(trainer.metrics_recorder, f)

    print('Start final testing')
    if args.test_weight is None:
        trainer.load_checkpoint(keyword='best')
        # note: also do self update here (indeed, the best practice might be doing self update before saving checkpoint)
        trainer.model.self_update()
    else:
        trainer.load_checkpoint_absolute(args.test_weight)
        # note: do self  update here
        trainer.model.self_update()

    trainer.val_loader = get_eval_loader(args.data_dir, args)
    trainer.progress_table = []
    testing_loss, _, stats, _ = trainer.validate([{}], final_testing=True)
    print('Final testing loss:', testing_loss)
    for key in stats:
        print(key, ':', stats[key])
    print('Finished')


def get_eval_loader(datadir, args):
    """
    Get the dataloader for validation and testing
    :param datadir: data root directory
    :param label_set: tasks
    :param args: arguments
    :param idx: The idx of the images in the current split (val_idx or test_idx)
    :return: The validation / testing dataloader
    """
    print(datadir)
    if 'nyu_v2' in datadir:
        val_dataset = NYU_v2(dataroot=datadir, mode='test')
    elif 'cityscapes' in datadir:
        val_dataset = CityScapes(dataroot=datadir, mode='test')
    else:
        raise NotImplementedError()

    print('Found', len(val_dataset), 'validation instances.')
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=None)
    return val_loader


program_start_time = time.time()


class Trainer:
    def __init__(self, train_loader, val_loader, model, optimizer, optimizer_W, scheduler, criteria, saver, args):
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.model = model
        self.optimizer = optimizer
        self.optimizer_W = optimizer_W
        self.scheduler = scheduler
        self.criteria = criteria
        self.args = args

        self.progress_table = []
        self.best_val_loss = 9e9
        self.stats = []
        self.start_epoch = 0
        self.loss_history = []
        self.saver = saver
        self.best_save_name = None

        self.epoch = -1
        self.mode = None  # the mode for self/co

        self.metrics_recorder = []

        print_table(self.progress_table, False)

    def train(self):
        for self.epoch in range(self.start_epoch, self.args.epochs):
            # train for one epoch
            self.train_prefetcher = DataPrefetcherMultiGPU(self.train_loader, devices=self.model.devices, tasks=self.model.tasks)
            train_string, train_stats = self.train_epoch()
            # evaluate on validation set
            progress_string = train_string
            loss, progress_string, val_stats, save_dict = self.validate(progress_string)
            self.saver.append_val(save_dict)
            self.checkpoint(loss)
            print()

            self.progress_table.append(progress_string)
            self.stats.append((train_stats, val_stats))

    def checkpoint(self, loss):
        """
        Save the model params if loss is the best val loss
        :param loss: current val loss
        """
        is_best = loss < self.best_val_loss
        self.best_val_loss = min(loss, self.best_val_loss)
        cur_path = self.args.save_dir + '/' + self.args.experiment_name + '/pretrained/'
        if not os.path.exists(cur_path):
            os.mkdir(cur_path)

        if self.args.save and self.epoch % 100 == 99:
            torch.save(self.model.state_dict(),
                       cur_path + 'epoch' + str(self.epoch) + '_valloss' + str(loss) + '.pth.tar')
        if self.args.save and is_best:
            cur_best_name = cur_path + 'best_epoch' + str(self.epoch) + '_valloss' + str(loss) + '.pth.tar'
            if self.best_save_name is not None:
                os.remove(self.best_save_name)
            self.best_save_name = cur_best_name
            torch.save(self.model.state_dict(), self.best_save_name)

    def load_checkpoint(self, keyword):
        cur_path = self.args.save_dir + '/' + self.args.experiment_name + '/pretrained/'
        found = False
        for checkpoint in os.listdir(cur_path):
            if keyword in checkpoint:
                print('Loading checkpoint from:')
                print(os.path.join(cur_path, checkpoint))
                checkpoint_file = torch.load(os.path.join(cur_path, checkpoint))
                self.model.load_state_dict(checkpoint_file)
                print("=> loaded checkpoint '{}'".format(os.path.join(cur_path, checkpoint)))
                found = True
                break
        if not found:
            raise RuntimeError('Checkpoint not found in', self.args.save_dir, 'with keyword:', keyword)

    def load_checkpoint_absolute(self, weight):
        print('Loading checkpoint from:')
        print(weight)
        checkpoint_file = torch.load(weight)
        self.model.load_state_dict(checkpoint_file)
        print("=> loaded checkpoint '{}'".format(weight))

    def train_epoch(self):
        global program_start_time
        average_meters = defaultdict(AverageMeter)
        display_values = []
        for name, func in self.criteria.items():
            display_values.append(name)

        # switch to train mode
        self.model.train()

        epoch_start_time = time.time()
        epoch_start_time2 = time.time()
        batch_num = 0
        num_data_points = len(self.train_loader) // self.args.virtual_batch_multiplier

        if self.args.epoch_wise:
            if self.epoch % (self.args.self_training_count + self.args.co_training_count) == self.args.self_training_count:
                self.mode = 'co-training'
                for name, param in self.model.named_parameters():
                    if name.startswith('W_networks'):
                        param.requires_grad = True
                    else:
                        if self.args.no_em:
                            param.requires_grad = True
                        else:
                            param.requires_grad = False
            elif self.epoch % (self.args.self_training_count + self.args.co_training_count) == 0:
                self.mode = 'self-training'
                for name, param in self.model.named_parameters():
                    if name.startswith('W_networks'):
                        param.requires_grad = False
                    else:
                        param.requires_grad = True

        while True:
            if batch_num == 0:
                epoch_start_time2 = time.time()
            if num_data_points == batch_num:
                break
            self.percent = batch_num / num_data_points
            loss_dict = None
            loss = 0

            ''' The key part in training starts here (refer to our proposed two-stage training algorithm) '''
            # set up models before training inference
            if not self.args.epoch_wise:
                # first use this condition, for the 0 vs x case
                if batch_num % (self.args.self_training_count + self.args.co_training_count) == self.args.self_training_count:
                    self.mode = 'co-training'
                    for name, param in self.model.named_parameters():
                        if name.startswith('W_networks'):
                            param.requires_grad = True
                        else:
                            if self.args.no_em:
                                param.requires_grad = True
                            else:
                                param.requires_grad = False
                elif batch_num % (self.args.self_training_count + self.args.co_training_count) == 0:
                    self.mode = 'self-training'
                    for name, param in self.model.named_parameters():
                        if name.startswith('W_networks'):
                            param.requires_grad = False
                        else:
                            param.requires_grad = True

            # accumulate gradients over multiple runs of input (virtual batch)
            for _ in range(self.args.virtual_batch_multiplier):
                data_start = time.time()
                input, target = self.train_prefetcher.next()
                average_meters['data_time'].update(time.time() - data_start)
                loss_dict2, loss2 = self.train_batch(input, target, mode=self.mode)
                loss += loss2
                if loss_dict is None:
                    loss_dict = loss_dict2
                else:
                    for key, value in loss_dict2.items():
                        loss_dict[key] += value

            # divide by the number of accumulations
            loss /= self.args.virtual_batch_multiplier
            for key, value in loss_dict.items():
                loss_dict[key] = value / self.args.virtual_batch_multiplier

            # save in saver
            self.saver.append(loss_dict)

            if self.mode == 'self-training':
                self.optimizer.step()
                self.optimizer.zero_grad()
            elif self.mode == 'co-training':
                self.optimizer_W.step()
                self.optimizer_W.zero_grad()
                if self.args.no_em:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            else:
                raise RuntimeError()

            if not self.args.epoch_wise:
                if batch_num % (self.args.self_training_count + self.args.co_training_count) == \
                        self.args.self_training_count + self.args.co_training_count - 1 or batch_num == num_data_points - 1:
                    self.model.self_update()
                    self.mode = 'self-training'

            ''' The key part in training ends here '''

            self.loss_history.append(float(loss))

            for name, value in loss_dict.items():
                try:
                    average_meters[name].update(value.data)
                except:
                    average_meters[name].update(value)

            elapsed_time_for_epoch = (time.time() - epoch_start_time2)
            eta = (elapsed_time_for_epoch / (batch_num + .2)) * (num_data_points - batch_num)
            if eta >= 24 * 3600:
                eta = 24 * 3600 - 1
            batch_num += 1
            current_learning_rate = get_average_learning_rate(self.optimizer)

            # the following section is used to print logs (irrelevant to training process)
            to_print = {}
            to_print['ep'] = ('{0}:').format(self.epoch)
            to_print['#/{0}'.format(num_data_points)] = ('{0}').format(batch_num)
            to_print['md'] = 'S' if self.mode == 'self-training' else 'C'
            to_print['lr'] = ('{0:0.4g}').format(current_learning_rate)
            to_print['eta'] = ('{0}').format(time.strftime("%H:%M:%S", time.gmtime(int(eta))))
            to_print['d%'] = ('{0:0.2g}').format(100 * average_meters['data_time'].sum / elapsed_time_for_epoch)
            for name in display_values:
                meter = average_meters[name]
                to_print[name] = ('{meter.avg:.4g}').format(meter=meter)
            if batch_num < num_data_points - 1:
                to_print['ETA'] = ('{0}').format(
                    time.strftime("%H:%M:%S", time.gmtime(int(eta + elapsed_time_for_epoch))))
            if batch_num % self.args.print_frequency == 0:
                print_table(self.progress_table + [[to_print]])

        if self.args.epoch_wise:
            if self.mode == 'self-training':
                self.scheduler.step()
            if self.epoch % (self.args.self_training_count + self.args.co_training_count) == \
                    self.args.self_training_count + self.args.co_training_count - 1 or self.epoch == self.args.epochs - 1:
                self.model.self_update()
                self.mode = 'self-training'

        epoch_time = time.time() - epoch_start_time
        stats = {'batches': num_data_points,
                 'learning_rate': current_learning_rate,
                 'Epoch time': epoch_time,
                 }
        for name in display_values:
            meter = average_meters[name]
            stats[name] = meter.avg

        to_print['eta'] = ('{0}').format(time.strftime("%H:%M:%S", time.gmtime(int(epoch_time))))

        return [to_print], stats

    def train_batch(self, input, target, mode):
        loss_dict = {}
        for _ in input:
            _ = _.float()
        output = self.model(input, is_sep_learn=False if mode == 'co-training' else True)

        first_loss = 'Loss'
        loss_dict[first_loss] = self.criteria[first_loss](output, target)

        for c_name, criterion_fun in self.criteria.items():
            if c_name != first_loss:
                loss_dict[c_name] = criterion_fun(output, target)
        loss = loss_dict[first_loss].clone()
        loss = loss / self.args.virtual_batch_multiplier
        loss.backward()

        return loss_dict, loss

    def validate(self, train_table, final_testing=False):
        average_meters = defaultdict(AverageMeter)
        self.model.eval()
        epoch_start_time = time.time()
        batch_num = 0
        num_data_points = len(self.val_loader)
        prefetcher = DataPrefetcherMultiGPU(self.val_loader, devices=self.model.devices, tasks=self.model.tasks)
        torch.cuda.empty_cache()
        with torch.no_grad():
            if self.epoch % self.args.test_interval == self.args.test_interval - 1 or final_testing:
                records = {}
                batch_size = []
                val_metrics = {}
                if 's' in self.args.tasks:
                    assert (self.args.num_seg_cls != -1)
                    records['seg'] = {'mIoUs': [], 'pixelAccs': [], 'errs': [],
                                      'conf_mat': np.zeros((self.args.num_seg_cls, self.args.num_seg_cls)),
                                      'labels': np.arange(self.args.num_seg_cls)}
                if 'n' in self.args.tasks:
                    records['sn'] = {'cos_similaritys': []}
                if 'd' in self.args.tasks:
                    records['depth'] = {'abs_errs': [], 'rel_errs': [], 'sq_rel_errs': [], 'ratios': [], 'rms': [],
                                        'rms_log': []}

            for i in range(len(self.val_loader)):
                input, target = prefetcher.next()
                if batch_num == 0:
                    epoch_start_time2 = time.time()
                output = self.model(input, is_sep_learn=False if self.args.epoch_wise and self.mode == 'co-training' else True)
                loss_dict = {}
                first_loss = 'Loss'
                loss_dict[first_loss] = self.criteria[first_loss](output, target)

                for c_name, criterion_fun in self.criteria.items():
                    if c_name != first_loss:
                        loss_dict[c_name] = criterion_fun(output, target)
                batch_num = i + 1
                for name, value in loss_dict.items():
                    try:
                        average_meters[name].update(value.data)
                    except:
                        average_meters[name].update(value)
                eta = ((time.time() - epoch_start_time2) / (batch_num + .2)) * (len(self.val_loader) - batch_num)
                to_print = {}
                to_print['#/{0}'.format(num_data_points)] = ('{0}').format(batch_num)
                to_print['eta'] = ('{0}').format(time.strftime("%H:%M:%S", time.gmtime(int(eta))))
                for name in self.criteria.keys():
                    meter = average_meters[name]
                    to_print[name] = ('{meter.avg:.4g}').format(meter=meter)
                progress = train_table + [to_print]
                if batch_num % self.args.print_frequency == 0:
                    print_table(self.progress_table + [progress])

                if self.epoch % self.args.test_interval == self.args.test_interval - 1 or final_testing:
                    new_shape = input[0].shape[-2:]
                    if 's' in self.args.tasks:
                        s_output = F.interpolate(output['segment_semantic'], size=new_shape)
                    if 'n' in self.args.tasks:
                        n_output = F.interpolate(output['normal'], size=new_shape)
                    if 'd' in self.args.tasks:
                        d_output = F.interpolate(output['depth_zbuffer'], size=new_shape)

                    if 's' in self.args.tasks:
                        gt = target['segment_semantic'].view(-1)
                        labels = gt < self.args.num_seg_cls
                        gt = gt[labels].int()
                        # here, why need to permute?
                        # print('shape seg:', s_output.shape)
                        logits = s_output.permute(0, 2, 3, 1).contiguous().view(-1, self.args.num_seg_cls)
                        logits = logits[labels]
                        err = torch.nn.CrossEntropyLoss(ignore_index=-1 if 'cityscapes' in self.args.data_dir else 255)(logits, gt.long())
                        prediction = torch.argmax(s_output, dim=1)
                        prediction = prediction.unsqueeze(1)
                        # pixel acc
                        prediction = prediction.view(-1)
                        prediction = prediction[labels].int()
                        pixelAcc = (gt == prediction).float().mean()
                        pred, gt, pixelAcc, err = prediction.cpu().numpy(), gt.cpu().numpy(), pixelAcc.cpu().numpy(), err.cpu().numpy()
                        new_mat = confusion_matrix(gt, pred, records['seg']['labels'])
                        assert (records['seg']['conf_mat'].shape == new_mat.shape)

                        records['seg']['conf_mat'] += new_mat
                        records['seg']['pixelAccs'].append(pixelAcc)
                        records['seg']['errs'].append(err)
                    if 'n' in self.args.tasks:
                        prediction = n_output.permute(0, 2, 3, 1).contiguous().view(-1, 3)
                        gt = target['normal'].permute(0, 2, 3, 1).contiguous().view(-1, 3)
                        labels = gt.max(dim=1)[0] != 255
                        # if hasattr(self, 'normal_mask'):
                        #     gt_mask = self.normal_mask.permute(0, 2, 3, 1).contiguous().view(-1, 3)
                        #     labels = labels and gt_mask.int() == 1
                        gt = gt[labels]
                        prediction = prediction[labels]
                        gt = F.normalize(gt.float(), dim=1)
                        prediction = F.normalize(prediction, dim=1)
                        cos_similarity = torch.nn.CosineSimilarity()(gt, prediction).cpu().numpy()

                        records['sn']['cos_similaritys'].append(cos_similarity)
                    if 'd' in self.args.tasks:
                        binary_mask = (torch.sum(target['depth_zbuffer'], dim=1) > 3 * 1e-5).unsqueeze(1)
                        depth_output_true = d_output.masked_select(binary_mask)
                        depth_gt_true = target['depth_zbuffer'].masked_select(binary_mask)
                        abs_err = torch.abs(depth_output_true - depth_gt_true)
                        rel_err = torch.abs(depth_output_true - depth_gt_true) / depth_gt_true
                        sq_rel_err = torch.pow(depth_output_true - depth_gt_true, 2) / depth_gt_true
                        abs_err = (torch.sum(abs_err) / torch.nonzero(binary_mask).size(0)).cpu().numpy()
                        rel_err = (torch.sum(rel_err) / torch.nonzero(binary_mask).size(0)).cpu().numpy()
                        sq_rel_err = (torch.sum(sq_rel_err) / torch.nonzero(binary_mask).size(0)).cpu().numpy()
                        # calcuate the sigma
                        term1 = depth_output_true / depth_gt_true
                        term2 = depth_gt_true / depth_output_true
                        ratio = torch.max(torch.stack([term1, term2], dim=0), dim=0)[0].cpu().numpy()
                        # calcualte rms
                        rms = torch.pow(depth_output_true - depth_gt_true, 2).cpu().numpy()
                        rms_log = torch.pow(torch.log10(depth_output_true + 1e-7) - torch.log10(depth_gt_true + 1e-7), 2).cpu().numpy()

                        records['depth']['abs_errs'].append(abs_err)
                        records['depth']['rel_errs'].append(rel_err)
                        records['depth']['sq_rel_errs'].append(sq_rel_err)
                        records['depth']['ratios'].append(ratio)
                        records['depth']['rms'].append(rms)
                        records['depth']['rms_log'].append(rms_log)
                    batch_size.append(len(input))

            if self.epoch % self.args.test_interval == self.args.test_interval - 1 or final_testing:
                if 's' in self.args.tasks:
                    val_metrics['seg'] = {}
                    jaccard_perclass = []
                    for i in range(self.args.num_seg_cls):
                        if not records['seg']['conf_mat'][i, i] == 0:
                            jaccard_perclass.append(
                                records['seg']['conf_mat'][i, i] / (np.sum(records['seg']['conf_mat'][i, :]) +
                                                                    np.sum(records['seg']['conf_mat'][:, i]) -
                                                                    records['seg']['conf_mat'][i, i]))

                    val_metrics['seg']['mIoU'] = np.sum(jaccard_perclass) / len(jaccard_perclass)

                    val_metrics['seg']['Pixel Acc'] = (np.array(records['seg']['pixelAccs']) * np.array(
                        batch_size)).sum() / sum(
                        batch_size)

                    val_metrics['seg']['err'] = (np.array(records['seg']['errs']) * np.array(batch_size)).sum() / sum(
                        batch_size)

                if 'n' in self.args.tasks:
                    val_metrics['sn'] = {}
                    overall_cos = np.clip(np.concatenate(records['sn']['cos_similaritys']), -1, 1)

                    angles = np.arccos(overall_cos) / np.pi * 180.0
                    val_metrics['sn']['Angle Mean'] = np.mean(angles)
                    val_metrics['sn']['Angle Median'] = np.median(angles)
                    val_metrics['sn']['Angle 11.25'] = np.mean(np.less_equal(angles, 11.25)) * 100
                    val_metrics['sn']['Angle 22.5'] = np.mean(np.less_equal(angles, 22.5)) * 100
                    val_metrics['sn']['Angle 30'] = np.mean(np.less_equal(angles, 30.0)) * 100
                    val_metrics['sn']['Angle 45'] = np.mean(np.less_equal(angles, 45.0)) * 100

                if 'd' in self.args.tasks:
                    val_metrics['depth'] = {}
                    records['depth']['abs_errs'] = np.stack(records['depth']['abs_errs'], axis=0)
                    records['depth']['rel_errs'] = np.stack(records['depth']['rel_errs'], axis=0)
                    records['depth']['ratios'] = np.concatenate(records['depth']['ratios'], axis=0)
                    val_metrics['depth']['abs_err'] = (records['depth']['abs_errs'] * np.array(batch_size)).sum() / sum(
                        batch_size)
                    val_metrics['depth']['rel_err'] = (records['depth']['rel_errs'] * np.array(batch_size)).sum() / sum(
                        batch_size)
                    val_metrics['depth']['sigma_1.25'] = np.mean(np.less_equal(records['depth']['ratios'], 1.25)) * 100
                    val_metrics['depth']['sigma_1.25^2'] = np.mean(
                        np.less_equal(records['depth']['ratios'], 1.25 ** 2)) * 100
                    val_metrics['depth']['sigma_1.25^3'] = np.mean(
                        np.less_equal(records['depth']['ratios'], 1.25 ** 3)) * 100

                self.metrics_recorder.append(val_metrics)
                with open(self.args.save_dir + '/' + self.args.experiment_name + '/metrics.txt', 'a') as f:
                    if not final_testing:
                        to_write = '\n' + '***** Epoch: ' + str(self.epoch) + ' *****\n'
                    else:
                        to_write = '\n' + '***** (Final Testing) Epoch: ' + str(self.epoch) + ' *****\n'
                    for tk in val_metrics:
                        to_write = to_write + 'Summary for ' + tk + '\n'
                        for key in val_metrics[tk]:
                            to_write = to_write + key + ':' + str(val_metrics[tk][key]) + '\n'
                    f.write(to_write)
                    if final_testing:
                        print(to_write)

        epoch_time = time.time() - epoch_start_time
        stats = {'batches': len(self.val_loader),
                 'Epoch time': epoch_time,
                 }
        for name in self.criteria.keys():
            meter = average_meters[name]
            stats[name] = meter.avg
        ultimate_loss = stats['Loss']
        to_print['eta'] = ('{0}').format(time.strftime("%H:%M:%S", time.gmtime(int(epoch_time))))
        torch.cuda.empty_cache()
        save_dict = {_: stats[_] for _ in self.criteria.keys()}

        return float(ultimate_loss), progress, stats, save_dict


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
