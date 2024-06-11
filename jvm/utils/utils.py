import os.path as osp
import sys
from importlib import import_module


def cvt_moduleToDict(mod: sys.modules) -> dict:
    cfg = {
        name: value
        for name, value in mod.__dict__.items() if not name.startswith('__')
    }

    return cfg


def cvt_cfgPathToDict(path: str) -> dict:

    abs_path = osp.abspath(path)

    sys.path.append(osp.split(abs_path)[0])
    _mod = import_module(osp.split(abs_path)[1].replace('.py', ''))

    return cvt_moduleToDict(_mod)


def check_cfg(cfg: dict) -> None:

    if cfg['resume_from'] is not None and cfg['load_from'] is not None:
        raise ValueError(
            "Only one of resume_from and load_from can be specified.")

    if 'optimizer' in cfg and 'optimizer_chain' in cfg:
        raise ValueError(
            "Only one of optimizer and optimizer_chain can be specified.")
