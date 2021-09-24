import random
from os import PathLike
from pathlib import Path
from typing import Any, List, Dict

import h5py
import numpy as np
import yaml

import cv2
import imageio

def get_ckpt_dir(model_dir: PathLike) -> Path:
    return Path(model_dir) / 'checkpoint'

def get_ckpt_path(model_dir: PathLike,
                  split_path: PathLike,
                  split_index: int) -> Path:
    split_path = Path(split_path)
    return get_ckpt_dir(model_dir) / f'{split_path.name}.{split_index}.pt'

def load_yaml(path: PathLike) -> Any:
    with open(path) as f:
        obj = yaml.safe_load(f)
    return obj

def dump_yaml(obj: Any, path: PathLike) -> None:
    with open(path, 'w') as f:
        yaml.dump(obj, f)

def scale(X, x_min, x_max):
    nom = (X-X.min(axis=0))*(x_max-x_min)
    denom = X.max(axis=0) - X.min(axis=0)
    denom[denom==0] = 1
    return x_min + nom/denom 