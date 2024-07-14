import os
import sys
import json
import time
import errno
import numpy as np
import random
import os.path as osp
import warnings
from collections import defaultdict
import torch.distributed as dist

import torch.utils.data as data


def mkdir_if_missing(dirname):
    """Create dirname if it is missing."""
    if not osp.exists(dirname):
        try:
            os.makedirs(dirname)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


def read_splits(filepath, path_prefix):
    def _convert(items):
        return [(os.path.join(path_prefix, impath), label, classname)
                for impath, label, classname in items]

    print(f"Reading split from {filepath}")
    data = json.load(open(filepath, "r"))
    train = _convert(data["train"])
    val = _convert(data["val"])
    test = _convert(data["test"])

    return train, val, test


def read_and_split_data(image_dir, p_trn=0.5, p_val=0.2, ignored=(), new_cnames=None):
    # The data are supposed to be organized into the following structure
    # =============
    # images/
    #     dog/
    #     cat/
    #     horse/
    # =============
    categories = sorted([f for f in os.listdir(image_dir) if not f.startswith(".")])
    categories = [c for c in categories if c not in ignored]
    categories.sort()

    p_tst = 1 - p_trn - p_val
    print(f"Splitting into {p_trn:.0%} train, {p_val:.0%} val, and {p_tst:.0%} test")

    def _collate(ims, y, c):
        return [(im, y, c) for im in ims]

    train, val, test = [], [], []
    for label, category in enumerate(categories):
        category_dir = os.path.join(image_dir, category)
        images = sorted([f for f in os.listdir(category_dir) if not f.startswith(".")])
        images = [os.path.join(category_dir, im) for im in images]
        random.shuffle(images)
        n_total = len(images)
        n_train = round(n_total * p_trn)
        n_val = round(n_total * p_val)
        n_test = n_total - n_train - n_val
        assert n_train > 0 and n_val > 0 and n_test > 0

        if new_cnames is not None and category in new_cnames:
            category = new_cnames[category]

        train.extend(_collate(images[:n_train], label, category))
        val.extend(_collate(images[n_train : n_train + n_val], label, category))
        test.extend(_collate(images[n_train + n_val :], label, category))

    return train, val, test


def generate_fewshot_dataset(data, num_shots=-1, repeat=False, seed=None):
    """Generate a few-shot dataset (typically for the training set).
    This function is useful when one wants to evaluate a model
    in a few-shot learning setting where each class only contains
    a small number of images.
    Args:
        data_sources: each individual is a list containing Datum objects.
        num_shots (int): number of instances per class to sample.
        repeat (bool): repeat images if needed (default: False).
    """
    if num_shots < 1:
        return data
    if seed is not None:
        random.seed(seed)

    cls2files = split_dataset_by_label(data)

    db = []
    for label, items in cls2files.items():
        if len(items) >= num_shots:
            sampled_items = random.sample(items, num_shots)
        else:
            sampled_items = random.choices(items, k=num_shots) if repeat else items
        db.extend(sampled_items)

    return db


def split_dataset_by_label(data_source):
    """Split a dataset, i.e. a list of Datum objects,
    into class-specific groups stored in a dictionary.
    Args:
        data_source (list): a list of Datum objects.
    """
    output = defaultdict(list)

    for item in data_source:
        output[item[2]].append(item)

    return output


def create_loader(dataset, batch_size, num_workers=0, num_tasks=1, global_rank=0, pin_memory=True, drop_last=True, persistent_workers=False):
    if dist.is_available() and dist.is_initialized():
        sampler_train = data.DistributedSampler(
            dataset, num_replicas=num_tasks, rank=global_rank, shuffle=True)
    else:
        sampler_train = data.RandomSampler(dataset)

    loader = data.DataLoader(
        dataset,
        sampler=sampler_train,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        persistent_workers=persistent_workers,
    )
    return loader


class ImageListDataset(data.Dataset):
    def __init__(self, image_list, label_list, class_desc, transform=None):
        """TODO: to be defined.
        :pair_filelist: TODO
        """
        from torchvision.datasets.folder import default_loader
        self.loader = default_loader
        data.Dataset.__init__(self)
        self.samples = [(fn, lbl) for fn, lbl in zip(image_list, label_list)]
        self.transform = transform
        self.class_desc = class_desc

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        desc = self.class_desc[target]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, target

    def __len__(self) -> int:
        return len(self.samples)