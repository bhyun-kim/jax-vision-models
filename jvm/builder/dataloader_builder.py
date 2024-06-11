from copy import deepcopy
from typing import Callable

import numpy as np
import torchvision as tv
from jax import numpy as jnp
from torch.utils.data import DataLoader


def collate_jnp(batch: list) -> tuple:

    data, target = zip(*batch)

    data = np.array(data)
    data = jnp.transpose(data, (0, 2, 3, 1))
    target = jnp.array(target)

    return data, target


def build_dataloader(dataloader_cfg: dict) -> DataLoader:

    _dataloader_cfg = deepcopy(dataloader_cfg)

    dataset_cfg = _dataloader_cfg.pop("dataset")

    transforms_cfg = dataset_cfg.pop("transforms")
    transforms = build_transforms(transforms_cfg)
    dataset_cfg["transform"] = transforms

    name = dataset_cfg.pop('name')
    dataset = getattr(tv.datasets, name)(**dataset_cfg)

    return DataLoader(dataset=dataset,
                      collate_fn=collate_jnp,
                      **dataloader_cfg['dataloader'])


def build_transforms(transform_cfgs: list) -> Callable:

    _transform_cfg = deepcopy(transform_cfgs)

    transforms = []
    for t_cfg in _transform_cfg:

        name = t_cfg.pop('name')
        transforms.append(getattr(tv.transforms, name)(**t_cfg))

    return tv.transforms.Compose(transforms)
