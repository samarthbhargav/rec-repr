# based on https://github.com/dawenl/vae_cf

from scipy import sparse
import pandas as pd
import os
import torch

import logging
from src import utils

log = logging.getLogger(__name__)


def sparse2torch(data):
    """
    From: https://discuss.pytorch.org/t/sparse-tensor-use-cases/22047/2
    :param data: a scipy sparse csr matrix
    :return: a sparse torch tensor
    """
    samples = data.shape[0]
    features = data.shape[1]
    values = data.data
    coo_data = data.tocoo()
    indices = torch.LongTensor([coo_data.row, coo_data.col])
    t = torch.sparse.FloatTensor(indices, torch.from_numpy(
        values).float(), [samples, features])
    return t


def load_train_data(csv_file, n_items):
    tp = pd.read_csv(csv_file)
    tp.sort_values(by="uid", inplace=True)

    # original_id -> data index
    oid2didx = {}
    # data index -> original_id
    didx2oid = {}

    rows, cols, data = [], [], []
    for uid, gr in tp.groupby("uid"):
        assert uid not in oid2didx
        did = len(oid2didx)
        oid2didx[uid] = did
        didx2oid[did] = uid
        rows.extend([did] * len(gr))
        cols.extend(gr["sid"].values)
        data.extend([1] * len(gr))

    data = sparse.csr_matrix((data, (rows, cols)),
                             shape=(len(didx2oid), n_items), dtype="float64")

    return data, oid2didx, didx2oid


def load_tr_te_data(csv_file_tr, csv_file_te, n_items):
    tp_tr = pd.read_csv(csv_file_tr)
    tp_tr.sort_values(by="uid", inplace=True)

    # original_id -> data index
    oid2didx = {}
    # data index -> original_id
    didx2oid = {}

    rows, cols, data = [], [], []
    for uid, gr in tp_tr.groupby("uid"):
        assert uid not in oid2didx
        did = len(oid2didx)
        oid2didx[uid] = did
        didx2oid[did] = uid
        rows.extend([did] * len(gr))
        cols.extend(gr["sid"].values)
        data.extend([1] * len(gr))

    assert len(rows) == len(cols) == len(data) == len(tp_tr)

    inp_mat = sparse.csr_matrix((data, (rows, cols)), shape=(len(didx2oid), n_items), dtype="float64")

    tp_te = pd.read_csv(csv_file_te)
    tp_te.sort_values(by="uid", inplace=True)
    rows, cols, data = [], [], []
    for uid, gr in tp_te.groupby("uid"):
        did = oid2didx[uid]
        rows.extend([did] * len(gr))
        cols.extend(gr["sid"].values)
        data.extend([1] * len(gr))

    assert len(rows) == len(cols) == len(data) == len(tp_te)
    eval_mat = sparse.csr_matrix((data, (rows, cols)), shape=(len(didx2oid), n_items), dtype="float64")

    assert inp_mat.shape[0] == eval_mat.shape[0] == len(oid2didx) == len(didx2oid)

    return inp_mat, eval_mat, oid2didx, didx2oid


class DataMgr:

    def __init__(self, dataset, cache_dir="./.data_cache"):
        self.dataset = dataset
        if dataset.startswith("gr"):
            self.data_root = os.path.join("datasets", dataset)
        else:
            self.data_root = os.path.join("datasets", dataset, "processed")

        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

    def get_data(self):


        unique_sid = list()
        with open(os.path.join(self.data_root, 'unique_sid.txt'), 'r') as f:
            for line in f:
                unique_sid.append(line.strip())

        unique_uid = list()
        with open(os.path.join(self.data_root, 'unique_uid.txt'), 'r') as f:
            for line in f:
                unique_uid.append(line.strip())

        n_items = len(unique_sid)
        n_users = len(unique_uid)

        train_data_cache = os.path.join(self.cache_dir, f"{self.dataset}_train.npz")
        if not os.path.exists(train_data_cache):
            log.info("Loading train data")
            train_data, train_oid2didx, train_didx2oid = load_train_data(os.path.join(self.data_root, 'train.csv'),
                                                                         n_items)
            sparse.save_npz(train_data_cache, train_data)
            utils.save_pickle(os.path.join(self.cache_dir, f"{self.dataset}_train_oid2didx.pkl"), train_oid2didx)
            utils.save_pickle(os.path.join(self.cache_dir, f"{self.dataset}_train_didx2oid.pkl"), train_didx2oid)
        else:
            log.info("Loading train data from cache")
            train_data = sparse.load_npz(train_data_cache)
            train_oid2didx = utils.load_pickle(os.path.join(self.cache_dir, f"{self.dataset}_train_oid2didx.pkl"))
            train_didx2oid = utils.load_pickle(os.path.join(self.cache_dir, f"{self.dataset}_train_didx2oid.pkl"))

        cached_val_data = os.path.join(self.cache_dir, f"{self.dataset}_val_tr.npz")

        if not os.path.exists(cached_val_data):
            log.info("Loading val data")
            vad_data_tr, vad_data_te, val_oid2didx, val_didx2oid = load_tr_te_data(
                os.path.join(self.data_root, 'validation_tr.csv'),
                os.path.join(self.data_root, 'validation_te.csv'),
                n_items)
            sparse.save_npz(cached_val_data, vad_data_tr)
            sparse.save_npz(os.path.join(self.cache_dir, f"{self.dataset}_val_te.npz"), vad_data_te)
            utils.save_pickle(os.path.join(self.cache_dir, f"{self.dataset}_val_oid2didx.pkl"), val_oid2didx)
            utils.save_pickle(os.path.join(self.cache_dir, f"{self.dataset}_val_didx2oid.pkl"), val_didx2oid)

        else:
            log.info("Loading val data from cache")
            vad_data_tr = sparse.load_npz(cached_val_data)
            vad_data_te = sparse.load_npz(os.path.join(self.cache_dir, f"{self.dataset}_val_te.npz"))
            val_oid2didx = utils.load_pickle(os.path.join(self.cache_dir, f"{self.dataset}_val_oid2didx.pkl"))
            val_didx2oid = utils.load_pickle(os.path.join(self.cache_dir, f"{self.dataset}_val_didx2oid.pkl"))

        cached_test_data = os.path.join(self.cache_dir, f"{self.dataset}_test_tr.npz")
        if not os.path.exists(cached_test_data):
            log.info("Loading test data")
            test_data_tr, test_data_te, test_oid2didx, test_didx2oid = load_tr_te_data(
                os.path.join(self.data_root, 'test_tr.csv'),
                os.path.join(self.data_root, 'test_te.csv'),
                n_items)

            sparse.save_npz(cached_test_data, test_data_tr)
            sparse.save_npz(os.path.join(self.cache_dir, f"{self.dataset}_test_te.npz"), test_data_te)
            utils.save_pickle(os.path.join(self.cache_dir, f"{self.dataset}_test_oid2didx.pkl"), test_oid2didx)
            utils.save_pickle(os.path.join(self.cache_dir, f"{self.dataset}_test_didx2oid.pkl"), test_didx2oid)
        else:
            log.info("Loading test data from cache")
            test_data_tr = sparse.load_npz(cached_test_data)
            test_data_te = sparse.load_npz(os.path.join(self.cache_dir, f"{self.dataset}_test_te.npz"))
            test_oid2didx = utils.load_pickle(os.path.join(self.cache_dir, f"{self.dataset}_test_oid2didx.pkl"))
            test_didx2oid = utils.load_pickle(os.path.join(self.cache_dir, f"{self.dataset}_test_didx2oid.pkl"))

        data = train_data, vad_data_tr, vad_data_te, test_data_tr, test_data_te
        data = (x.astype('float32') for x in data)

        return data
