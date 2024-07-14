# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import builtins
import copy
import datetime
import os
import time
from collections import defaultdict, deque
from pathlib import Path

import torch
import torch.nn as nn
import torch.distributed as dist
from torch import inf


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=1, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, n=1, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v, n=n)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None, start_iter=0):
        i = start_iter
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))

    def log(self, iterator, header=None):
        if not header:
            header = ''
        space_fmt = ':' + str(len(str(iterator.num_iters))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            '{meters}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        if iterator.is_print_step():
            if torch.cuda.is_available():
                print(log_msg.format(
                    iterator.current_it, iterator.num_iters,
                    meters=str(self),
                    memory=torch.cuda.max_memory_allocated() / MB), flush=True)
            else:
                print(log_msg.format(
                    iterator.current_it, iterator.num_iters,
                    meters=str(self)), flush=True)


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    builtin_print = builtins.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        force = force or (get_rank() % 8 == 0)
        if is_master or force:
            now = datetime.datetime.now().time()
            builtin_print('[{}] '.format(now), end='')  # print with time stamp
            builtin_print(*args, **kwargs)

    builtins.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(local_rank, args):
    ngpus_per_node = torch.cuda.device_count()
    args.env.distributed = ngpus_per_node > 1 or args.env.world_size > 1
    if args.env.distributed:
        args.env.world_size = ngpus_per_node * args.env.world_size
        args.env.rank = args.env.rank * ngpus_per_node + local_rank
    else:
        print('Not using distributed mode')
        setup_for_distributed(is_master=True)  # hack
        args.env.world_size = 1
        args.env.rank = 0
        return

    print(args.env.dist_url, args.env.world_size, args.env.rank, flush=True)
    dist.init_process_group(backend='nccl', init_method=args.env.dist_url,
                            world_size=args.env.world_size, rank=args.env.rank)

    torch.cuda.set_device(local_rank)
    print('Distributed init (rank {}): {}, gpu {}'.format(
        args.env.rank, args.env.dist_url, local_rank), flush=True)
    torch.distributed.barrier()
    setup_for_distributed(args.env.rank == 0)


class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                # norm = get_grad_norm_(parameters)
                norm = None
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)


class NativeScalerWithMultipleOpt:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(self, loss, optimizers, clip_grad=None, parameters=None, create_graph=False, update_grad=True, step_opt=0):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                for opt in optimizers:
                    self._scaler.unscale_(opt)  # unscale the gradients of optimizer's assigned params in-place
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                for opt in optimizers:
                    self._scaler.unscale_(opt)
                # norm = get_grad_norm_(parameters)
                norm = None
            self._scaler.step(optimizers[step_opt])
            self._scaler.update()

            for opt in optimizers:
                opt.zero_grad()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)


def get_grad_norm_(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
    return total_norm


def save_model(args, epoch, model, model_without_ddp, optimizer, loss_scaler):
    output_dir = Path(args.output_dir)
    epoch_name = str(epoch)
    if loss_scaler is not None:
        checkpoint_paths = [output_dir / ('checkpoint-%s.pth' % epoch_name)]
        for checkpoint_path in checkpoint_paths:
            to_save = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'scaler': loss_scaler.state_dict(),
                'args': args,
            }

            save_on_master(to_save, checkpoint_path)
    else:
        client_state = {'epoch': epoch}
        model.save_checkpoint(save_dir=args.output_dir, tag="checkpoint-%s" % epoch_name, client_state=client_state)


def load_model(args, model_without_ddp, optimizer, loss_scaler):
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        print("Resume checkpoint %s" % args.resume)
        if 'optimizer' in checkpoint and 'epoch' in checkpoint and not (hasattr(args, 'eval') and args.eval):
            optimizer.load_state_dict(checkpoint['optimizer'])
            args.start_epoch = checkpoint['epoch'] + 1
            if 'scaler' in checkpoint:
                loss_scaler.load_state_dict(checkpoint['scaler'])
            print("With optim & sched!")


def all_reduce_mean(x):
    world_size = get_world_size()
    if world_size > 1:
        x_reduce = torch.tensor(x).cuda()
        dist.all_reduce(x_reduce)
        x_reduce /= world_size
        return x_reduce.item()
    else:
        return x

@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    if not torch.distributed.is_initialized():
        return tensor
    tensors_gather = [torch.ones_like(tensor)
                      for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


@torch.no_grad()
def eval_knn(eval_loader, model, epoch, args, device):
    from torch.nn import functional as F
    from copy import deepcopy
    print(f'==> Begin evaluation epoch {epoch}')
    metric_logger = MetricLogger(delimiter="  ")

    model = deepcopy(model.module if args.env.distributed else model)
    model.encoder.norm = nn.Identity()
    model.encoder.head = nn.Identity()
    model.train(False)

    # Extract features
    features, labels = [], []
    print_freq = args.log.print_freq
    for images, y in metric_logger.log_every(eval_loader, print_freq, 'Extract features'):
        images, y = images.to(device, non_blocking=True), y.to(device, non_blocking=True)
        features.append(model.encoder(images)[:, 1:].mean(1))
        labels.append(y)

    # Synchronize across gpus
    features = concat_all_gather(F.normalize(torch.cat(features), p=2, dim=1))
    labels = concat_all_gather(torch.cat(labels))

    # kNN Evaluation
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('Acc1', SmoothedValue(fmt='avg:6.3f'))
    for i in metric_logger.log_every(range(0, features.shape[0], args.batch_size), 250):
        qfeats = features[i:i+args.batch_size]
        qlbls = labels[i:i+args.batch_size]

        scores = torch.einsum('qd,nd->qn', qfeats, features)
        topk_idx = torch.topk(scores, k=2, dim=1, sorted=True)[1][:, 1]
        topk_lbl = labels[topk_idx]

        acc1 = (topk_lbl == qlbls).float().mean()*100
        metric_logger.update(Acc1=acc1, n=qlbls.shape[0])

    metric_logger.synchronize_between_processes()
    print(f"NN Acc1: {metric_logger.meters['Acc1'].global_avg:6.3f}")
    return metric_logger.meters['Acc1'].global_avg



def init_wandb(args, job_dir, entity, project, job_name):
    import wandb
    wandb_dir = os.path.join(job_dir, 'wandb')
    os.makedirs(wandb_dir, exist_ok=True)
    runid = None
    if os.path.exists(f"{wandb_dir}/runid.txt"):
        runid = open(f"{wandb_dir}/runid.txt").read()
    wandb.init(project=project,
               name=job_name,
               dir=wandb_dir,
               entity=entity,
               resume="allow",
               id=runid)
    open(f"{wandb_dir}/runid.txt", 'w').write(wandb.run.id)
    wandb.config.update({k: args[k] for k in args if k not in wandb.config})


class CheckpointManager:
    def __init__(self,
                 modules,
                 ckpt_dir,
                 epochs,
                 save_freq=None,
                 suffix=''):
        self.modules = modules
        self.ckpt_dir = ckpt_dir
        self.epochs = epochs
        self.save_freq = save_freq
        self.suffix = suffix
        self.state_queue = deque(maxlen=2)

        self.distributed = dist.is_available() and dist.is_initialized()
        self.world_size = dist.get_world_size() if self.distributed else 1
        self.rank = dist.get_rank() if self.distributed else 0

        if self.rank == 0:
            os.makedirs(os.path.join(self.ckpt_dir), exist_ok=True)

    def resume(self):
        ckpt_fname = os.path.join(self.ckpt_dir, f'checkpoint_latest{self.suffix}.pth')
        start_epoch = 0
        if os.path.isfile(ckpt_fname):
            checkpoint = torch.load(ckpt_fname, map_location='cpu')

            # Load state dict
            for k in self.modules:
                try:
                    msg = self.modules[k].load_state_dict(checkpoint[k], strict=False)
                    assert len(msg.missing_keys) == 0, f"Missing keys: {msg.missing_keys}"
                except:
                    self.modules[k].load_state_dict(checkpoint[k])
            start_epoch = checkpoint['epoch']
            print(f"=> loaded checkpoint '{ckpt_fname}' (epoch {checkpoint['epoch']})")

        return start_epoch

    def resume_v2(self):
        ckpt_fname = os.path.join(self.ckpt_dir, f'checkpoint_latest{self.suffix}.pth')
        if os.path.isfile(ckpt_fname):
            checkpoint = torch.load(ckpt_fname, map_location='cpu')

            # Load state dict
            for k in self.modules:
                self.modules[k].load_state_dict(checkpoint[k])
            metrics = {k: checkpoint[k] for k in checkpoint if k not in self.modules}
            print(f"=> Loaded checkpoint: '{ckpt_fname}'.\n * {str(metrics)}")
            return metrics

    def create_state_dict(self, save_dict):
        state = {k: self.modules[k].state_dict() for k in self.modules}
        if save_dict is not None:
            state.update(save_dict)
        return state

    def load_state_dict(self, checkpoint):
        for k in self.modules:
            self.modules[k].load_state_dict(checkpoint[k])
        metrics = {k: checkpoint[k] for k in checkpoint if k not in self.modules}
        return metrics

    def checkpoint(self, epoch, save_dict=None):
        if self.rank != 0:
            return
        state = self.create_state_dict(save_dict)
        ckpt_fname = os.path.join(self.ckpt_dir, f'checkpoint_latest{self.suffix}.pth')
        torch.save(state, ckpt_fname)
        print(f"=> saved checkpoint '{ckpt_fname}' (epoch {epoch})")

        if self.save_freq is not None and ((epoch % self.save_freq == 0) or epoch == self.epochs):
            ckpt_fname = os.path.join(self.ckpt_dir, f'checkpoint_{epoch:04d}{self.suffix}.pth')
            torch.save(state, ckpt_fname)
            print(f"=> saved checkpoint '{ckpt_fname}' (epoch {epoch})")

    def checkpoint_v2(self, identifier='latest', save_dict=None):
        if self.rank != 0:
            return
        state = self.create_state_dict(save_dict)
        ckpt_fname = os.path.join(self.ckpt_dir, f'checkpoint_{identifier}{self.suffix}.pth')
        torch.save(state, ckpt_fname)
        print(f"=> saved checkpoint '{ckpt_fname}'")

    def store_state(self, iteration):
        self.state_queue.append({
            'iter': iteration,
            'state': copy.deepcopy(self.create_state_dict(None))
        })

    def restore_state(self):
        if len(self.state_queue) == 2:
            self.state_queue.pop()
            state = self.state_queue.pop()
            self.load_state_dict(state['state'])
            return state['iter']
        elif len(self.state_queue) == 1:
            self.state_queue.pop()
        return -1
