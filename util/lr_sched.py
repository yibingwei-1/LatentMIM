# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate with half-cycle cosine after warmup"""
    min_lr = 0.
    if 'min_lr_frac' in dict(args):
        min_lr = args.lr * args.min_lr_frac
    if epoch < args.warmup_epochs:
        lr = min_lr + (args.lr - min_lr) * epoch / args.warmup_epochs
    else:
        lr = min_lr + (args.lr - min_lr) * 0.5 * (1. + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
        if "lr_scale" in param_group:
            param_group["lr"] *= param_group["lr_scale"]
    return lr
