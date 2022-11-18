import pickle as pkl

import os
import logging
import random
from typing import Dict

import numpy as np
import torch
import gzip
import json
import time
from logging import handlers
import json

class Timer:
    def __init__(self, name=None):
        self.name = name
        self.start = None

    def __enter__(self):
        self.start = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        ps = ""
        if self.name:
            ps += f"{self.name}: "
        ps += f"{time.time() - self.start: .4f}s"
        print(ps)

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        # if isinstance(obj, np.ndarray):
        #     return obj.tolist()
        return super(NpEncoder, self).default(obj)


def set_seed(seed):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def supress_log(lib_name):
    logging.getLogger(lib_name).setLevel(logging.INFO)


def add_common_args(parser):
    parser.add_argument(
        "--verbose", help="If set, show verbose logs", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=42)


def configure_logging(module, verbose, log_dir="./logs"):
    os.makedirs(log_dir, exist_ok=True)
    handlers = [
        logging.handlers.RotatingFileHandler(
            f"{log_dir}/{module}.log", maxBytes=1048576 * 5, backupCount=7),
        logging.StreamHandler()
    ]
    log_format = "[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s"
    if verbose:
        logging.basicConfig(level=logging.DEBUG,
                            handlers=handlers, format=log_format)
    else:
        logging.basicConfig(level=logging.INFO,
                            handlers=handlers, format=log_format)


def load_pickle(path: str) -> object:
    with open(path, "rb") as reader:
        return pkl.load(reader)


def save_pickle(path: str, obj: object):
    with open(path, "wb") as writer:
        pkl.dump(obj, writer)


def write_json(d: Dict, path: str, indent=None, zipped=False):
    if zipped:
        with gzip.open(path, 'wt', encoding="ascii") as zipfile:
            json.dump(d, zipfile, indent=indent, cls=NpEncoder)
    else:
        with open(path, "w") as writer:
            json.dump(d, writer, indent=indent, cls=NpEncoder)


def read_json(path, zipped=False):
    if zipped:
        with gzip.open(path, 'rt', encoding="ascii") as zipfile:
            return json.load(zipfile)
    else:
        with open(path, "r") as reader:
            return json.load(reader)


def read_file(path):
    with open(path, "r") as reader:
        return reader.read()
