import os
import torch.utils.data as data
import torchvision.datasets as datasets
import data_tools

__all__ = ['imagenet', 'imagenet100']


NUM_CLASSES = {
    'imagenet': 1000,
    'imagenet100': 100
}

class ImageListDataset(data.Dataset):
    def __init__(self, image_list, label_list, class_desc=None, transform=None):
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
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, target

    def __len__(self) -> int:
        return len(self.samples)


def imagenet(data_path, transform, train=True):
    import scipy.io
    meta = scipy.io.loadmat(os.path.join(data_path, 'anno', 'meta.mat'))['synsets']
    synsets = [m[0][1][0] for m in meta]
    descriptions = {m[0][1][0]: m[0][2][0] for m in meta}
    imagenet_ids = {m[0][1][0]: m[0][0][0][0]-1 for m in meta}
    if train:
        dataset = datasets.ImageFolder(os.path.join(data_path, 'train'), transform=transform)
        wnids = [fn.split('/')[-2] for fn, lbl in dataset.samples]
        inids = [imagenet_ids[w] for w in wnids]

        dataset.samples = [(fn, lbl) for (fn, _), lbl in zip(dataset.samples, inids)]
        dataset.imgs = dataset.samples
        dataset.targets = inids
    else:
        fns = [f"{data_path}/val/ILSVRC2012_val_{i:08d}.JPEG" for i in range(1,50001)]
        inids = [int(ln.strip())-1 for ln in open(os.path.join(data_path, 'anno', 'ILSVRC2012_validation_ground_truth.txt'))]

        dataset = ImageListDataset(fns, inids, transform=transform)
        dataset.imgs = dataset.samples
        dataset.targets = inids

    dataset.classes = synsets[:1000]
    dataset.descriptions = [descriptions[cls] for cls in dataset.classes]
    return dataset


def imagenet100(data_path, transform, train=True):
    # The list of classes from ImageNet-100, which are randomly sampled from the original ImageNet-1k dataset. This list can also be downloaded at: http://github.com/HobbitLong/CMC/blob/master/imagenet100.txt
    in100_cls_list = ['n02869837','n01749939','n02488291','n02107142','n13037406','n02091831','n04517823','n04589890','n03062245','n01773797','n01735189','n07831146','n07753275','n03085013','n04485082','n02105505','n01983481','n02788148','n03530642','n04435653','n02086910','n02859443','n13040303','n03594734','n02085620','n02099849','n01558993','n04493381','n02109047','n04111531','n02877765','n04429376','n02009229','n01978455','n02106550','n01820546','n01692333','n07714571','n02974003','n02114855','n03785016','n03764736','n03775546','n02087046','n07836838','n04099969','n04592741','n03891251','n02701002','n03379051','n02259212','n07715103','n03947888','n04026417','n02326432','n03637318','n01980166','n02113799','n02086240','n03903868','n02483362','n04127249','n02089973','n03017168','n02093428','n02804414','n02396427','n04418357','n02172182','n01729322','n02113978','n03787032','n02089867','n02119022','n03777754','n04238763','n02231487','n03032252','n02138441','n02104029','n03837869','n03494278','n04136333','n03794056','n03492542','n02018207','n04067472','n03930630','n03584829','n02123045','n04229816','n02100583','n03642806','n04336792','n03259280','n02116738','n02108089','n03424325','n01855672','n02090622']

    def filter_dataset(db, cls_list):
        cls2lbl = {cls: lbl for lbl, cls in enumerate(cls_list)}
        idx = [i for i, lbl in enumerate(db.targets) if db.classes[lbl] in cls2lbl]
        samples = [db.samples[i] for i in idx]
        db.samples = [(fn, cls2lbl[db.classes[lbl]]) for fn, lbl in samples]
        db.targets = [dt[1] for dt in db.samples]
        db.imgs = db.samples
        descriptions = {cls: desc for cls, desc in zip(db.classes, db.descriptions)}
        db.classes = cls_list
        db.descriptions = [descriptions[cls] for cls in cls_list]

    dataset = imagenet(data_path, transform, train=train)
    filter_dataset(dataset, in100_cls_list)
    return dataset

def load_dataset(dataset, path, transform, train=True):
    return globals()[dataset](path, transform, train)


from util import misc
from torch.utils import data
def create_loader(dataset, batch_size, num_workers=0, num_tasks=1, global_rank=0, pin_memory=True, drop_last=True):
    if misc.is_dist_avail_and_initialized():
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
    )
    return loader