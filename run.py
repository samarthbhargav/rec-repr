import math
import os
from typing import Dict

import numpy as np

import torch
from torch import optim, nn

from scipy import sparse
from copy import deepcopy
import logging
import functools
from datetime import datetime

from src import utils
from src.data import DataMgr
from src.dae import MultDAE
from src.vae import MultVAE

from src import eval_rec
from collections import defaultdict
import argparse

log = logging.getLogger(__name__)


def generate(batch_size, device, data_in, data_out=None, shuffle=False, samples_perc_per_epoch=1.):
    assert 0 < samples_perc_per_epoch <= 1

    total_samples = data_in.shape[0]
    samples_per_epoch = int(total_samples * samples_perc_per_epoch)
    if shuffle:
        idxlist = np.arange(total_samples)
        np.random.shuffle(idxlist)
        idxlist = idxlist[:samples_per_epoch]
    else:
        idxlist = np.arange(samples_per_epoch)

    for batch_num, st_idx in enumerate(range(0, samples_per_epoch, batch_size)):
        end_idx = min(st_idx + batch_size, samples_per_epoch)
        idx = idxlist[st_idx:end_idx]

        yield batch_num, Batch(device, idx, data_in, data_out)


class Batch:
    def __init__(self, device, idx, data_in, data_out=None):
        self._device = device
        self._idx = idx
        self._data_in = data_in
        self._data_out = data_out

    def get_idx(self):
        return self._idx

    def get_idx_to_dev(self):
        return torch.LongTensor(self.get_idx()).to(self._device)

    def get_ratings(self, is_out=False):
        data = self._data_out if is_out else self._data_in
        return data[self._idx]

    def get_ratings_to_dev(self, is_out=False):
        return torch.Tensor(
            self.get_ratings(is_out).toarray()
        ).to(self._device)


def make_metrics(metrics):
    for m in metrics:
        k = int(m.split("_")[-1])
        if m.startswith("ndcg_cut"):
            yield m, functools.partial(eval_rec.ndcg, k=k)
        elif m.startswith("recall_"):
            yield m, functools.partial(eval_rec.recall, k=k)


def evaluate_fast(ratings_in: sparse.csr_matrix, ratings_out, ratings_pred, metrics, fill_rated_with_inf=True):
    if fill_rated_with_inf:
        ratings_pred[ratings_in.nonzero()] = -np.inf
    res = {}
    for metric, fn in make_metrics(metrics):
        res[metric] = fn(ratings_pred, ratings_out)
    return res


def evaluate_slow(ratings_in, ratings_out, ratings_pred, metrics, fill_rated_with_inf=True):
    if fill_rated_with_inf:
        ratings_pred[ratings_in.nonzero()] = -np.inf

    evaluator = eval_rec.PyTrecEvaluator(metrics=metrics,
                                         holdout=ratings_out)

    return evaluator.aggregate(evaluator.compute_metrics(ratings_pred), aggregate="gather")


def evaluate_single_batch(model: nn.Module, batch, metrics, fill_rated_with_inf=True, fast=False):
    assert model.training is False
    ratings_in = batch.get_ratings_to_dev()
    ratings_out = batch.get_ratings(is_out=True)

    with torch.no_grad():
        z, ratings_pred = model.infer(ratings_in, return_z=True)
        ratings_pred = ratings_pred.cpu().detach().numpy()

    if fast:
        return evaluate_fast(batch.get_ratings(),
                             ratings_out,
                             ratings_pred,
                             metrics,
                             fill_rated_with_inf=fill_rated_with_inf)
    else:
        return evaluate_slow(batch.get_ratings(),
                             ratings_out,
                             ratings_pred,
                             metrics,
                             fill_rated_with_inf=fill_rated_with_inf)


def evaluate(model, data_in, data_out, metrics, device, samples_perc_per_epoch=1., batch_size=500, fast=False):
    model.eval()

    # metric -> [useri, userj, ...]
    results = defaultdict(list)
    total_eval_steps = math.ceil((data_in.shape[0] * samples_perc_per_epoch) / batch_size)
    fill_rated_with_inf = not (data_in is data_out)
    log.debug(f"Fill Inf: {fill_rated_with_inf}, Fast:{fast}")
    print_every = max(10, total_eval_steps // 10)
    for batch_num, batch in generate(batch_size=batch_size,
                                     device=device,
                                     data_in=data_in,
                                     data_out=data_out,
                                     samples_perc_per_epoch=samples_perc_per_epoch):

        if batch_num % print_every == 0 or batch_num + 1 == total_eval_steps:
            log.info(f"Batch: [{batch_num + 1}/{total_eval_steps}]")

        res = evaluate_single_batch(model,
                                    batch,
                                    metrics,
                                    fast=fast,
                                    fill_rated_with_inf=fill_rated_with_inf)

        for metric, values in res.items():
            results[metric].extend(values)

        # if batch_num > 10:
        #     break
    metric_means = {}
    for m, scores in results.items():
        # mean, std
        metric_means[m] = (np.mean(scores), np.std(scores))

    return metric_means


def run(model, opts, train_data, n_epochs, batch_size, start_train_step, device):
    model.train()
    current_train_step = start_train_step
    total_train_steps = math.ceil(train_data.shape[0] / batch_size)
    print_every = max(10, total_train_steps // 10)
    exts = defaultdict(list)

    for epoch in range(n_epochs):
        total_supervised = 0
        total = 0
        for batch_num, batch in generate(batch_size=batch_size,
                                         device=device, data_in=train_data, shuffle=True):
            ratings = batch.get_ratings_to_dev()

            for optimizer in opts:
                optimizer.zero_grad()

            total += int(ratings.size(0))
            loss, ext = model.compute_loss(ratings,
                                           train_step=current_train_step,
                                           n_train=train_data.shape[0])
            for k, v in ext.items():
                exts[k].append(v)
            exts["loss"].append(loss.item())
            exts["train_step"].append(current_train_step)

            current_train_step += 1

            loss.backward()

            if batch_num % print_every == 0 or batch_num + 1 == total_train_steps:
                log.info(
                    f"Batch: [{batch_num + 1}/{total_train_steps}] Loss: {loss.item():.4f} [n_sup: {total_supervised}/{total}]")

            for optimizer in opts:
                optimizer.step()

    return current_train_step, exts


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, choices=["msd", "ml-20m", "ml-1m", "gr", "gr-comics", "gr-children"],
                        required=True)

    parser.add_argument("--out", type=str, required=False, default=None)
    parser.add_argument("--save", type=str, required=False, default=None)

    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default="cuda")

    # model config path, a JSON
    parser.add_argument("--model_config", type=str, required=True)

    # train args
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--n_epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=500)

    # misc args
    parser.add_argument("--verbose", action="store_true", default=False)

    return parser


def get_model(model_config) -> nn.Module:
    model_cls = {
        "multvae": MultVAE,
        "multdae": MultDAE,
    }.get(model_config["model_name"])

    if model_cls is None:
        raise NotImplementedError(model_config["model_name"])

    return model_cls(**model_config)


if __name__ == '__main__':
    start_time = datetime.now()
    parser = get_parser()
    args = parser.parse_args()

    utils.configure_logging("run", args.verbose)
    utils.set_seed(args.seed)

    log.info(f"args: {vars(args)}")
    device = torch.device(args.device)

    data_mgr = DataMgr(args.dataset)
    train_data, valid_in_data, valid_out_data, test_in_data, test_out_data = data_mgr.get_data()
    n_items = train_data.shape[1]

    model_config = utils.read_json(args.model_config)
    model_config["n_items"] = n_items
    model = get_model(model_config)
    model_best = get_model(model_config)

    model = model.to(device)
    model_best = model_best.to(device)

    log.info(f"Initialized model: {model}")

    per_epoch = math.ceil(train_data.shape[0] / args.batch_size)
    total_steps = args.n_epochs * per_epoch
    log.info(f"batches per epoch: {per_epoch}")

    best_ndcg = -np.inf
    train_scores, valid_scores = [], []
    params = set(model.parameters())

    if model_config["model_name"] == "multdae":
        # MultDAE requires weight_decay
        optimizer = optim.Adam(params, lr=args.lr, weight_decay=model_config["lam"])
    else:
        optimizer = optim.Adam(params, lr=args.lr)

    train_eval_perc = .1
    val_eval_perc = 1.0

    train_step = 0

    scores = {
        "val": [],
        "train": [],
        "train_ext": defaultdict(list)
    }

    train_metrics = ["ndcg_cut_100"]
    val_metrics = ["ndcg_cut_100", "recall_50"]

    for epoch in range(args.n_epochs):
        log.info(f"Epoch: {epoch}")
        train_step, train_extra_metrics = run(model=model,
                                              opts=[optimizer],
                                              train_data=train_data,
                                              n_epochs=1,
                                              batch_size=args.batch_size,
                                              start_train_step=train_step,
                                              device=device)
        for k, v in train_extra_metrics.items():
            scores["train_ext"][k].extend(v)
        log.info(f"[{epoch}] Epoch complete. Total train steps: {train_step}")

        log.info(f"[{epoch}] Evaluating on train set [subset percentage={train_eval_perc * 100:.1f}%]")
        train_epoch_score = evaluate(model,
                                     data_in=train_data,
                                     data_out=train_data,
                                     metrics=train_metrics,
                                     device=device,
                                     samples_perc_per_epoch=train_eval_perc,
                                     batch_size=args.batch_size,
                                     fast=True)
        log.info(f"[{epoch}] Train evaluation complete!")

        scores["train"].append(train_epoch_score)

        log.info(f"[{epoch}] Evaluating on val set [subset percentage={val_eval_perc * 100:.1f}%]")

        val_epoch_score = evaluate(model,
                                   data_in=valid_in_data,
                                   data_out=valid_out_data,
                                   metrics=val_metrics,
                                   device=device,
                                   samples_perc_per_epoch=val_eval_perc,
                                   batch_size=args.batch_size,
                                   fast=True)

        log.info(f"[{epoch}] Val evaluation complete!")

        scores["val"].append(val_epoch_score)

        # mean, std -> need mean
        curr_train_score = train_epoch_score["ndcg_cut_100"][0]
        curr_score = val_epoch_score["ndcg_cut_100"][0]

        if curr_score > best_ndcg:
            best_ndcg = curr_score
            model_best.load_state_dict(deepcopy(model.state_dict()))

        log.info(f'[{epoch}] valid ndcg@100: {curr_score:.4f} | ' +
                 f'best valid: {best_ndcg:.4f} | train ndcg@100: {curr_train_score:.4f}')

    end_time = datetime.now()
    duration = end_time - start_time
    run_time = {
        "start_time": start_time.isoformat(sep=" "),
        "end_time": end_time.isoformat(sep=" "),
        "time_taken_mins": divmod(duration.total_seconds(), 60)[0]
    }

    log.info(f"time: {run_time}")
    if args.out is not None:
        out = {
            "scores": scores,
            "args": vars(args),
            "run_stats": run_time,
            "model_config": model_config
        }

        dirname = os.path.dirname(args.out)
        if len(dirname) > 0:
            os.makedirs(dirname, exist_ok=True)
        utils.write_json(out, args.out, zipped=True)
        log.info(f"Wrote output to {args.out}")

    if args.save is not None:
        dirname = os.path.dirname(args.save)
        if len(dirname) > 0:
            os.makedirs(dirname, exist_ok=True)
        log.info(f"Saving best model to {args.save}")
        torch.save(model_best.state_dict(), args.save)
