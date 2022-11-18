import bottleneck as bn
import numpy as np

from collections import defaultdict
from typing import List, Dict, Any

import numpy as np
import pytrec_eval


def ndcg(X_pred, heldout_batch, k=100):
    '''
    normalized discounted cumulative gain@k for binary relevance
    ASSUMPTIONS: all the 0's in heldout_data indicate 0 relevance
    '''
    batch_users = X_pred.shape[0]
    idx_topk_part = bn.argpartition(-X_pred, k, axis=1)
    topk_part = X_pred[np.arange(batch_users)[:, np.newaxis],
                       idx_topk_part[:, :k]]
    idx_part = np.argsort(-topk_part, axis=1)
    # X_pred[np.arange(batch_users)[:, np.newaxis], idx_topk] is the sorted
    # topk predicted score
    idx_topk = idx_topk_part[np.arange(batch_users)[:, np.newaxis], idx_part]
    # build the discount template
    tp = 1. / np.log2(np.arange(2, k + 2))

    DCG = (heldout_batch[np.arange(batch_users)[:, np.newaxis],
                         idx_topk].toarray() * tp).sum(axis=1)
    IDCG = np.array([(tp[:min(n, k)]).sum()
                     for n in heldout_batch.getnnz(axis=1)])
    return DCG / IDCG


def recall(X_pred, heldout_batch, k=100):
    batch_users = X_pred.shape[0]

    idx = bn.argpartition(-X_pred, k, axis=1)
    X_pred_binary = np.zeros_like(X_pred, dtype=bool)
    X_pred_binary[np.arange(batch_users)[:, np.newaxis], idx[:, :k]] = True

    X_true_binary = (heldout_batch > 0).toarray()
    tmp = (np.logical_and(X_true_binary, X_pred_binary).sum(axis=1)).astype(
        np.float32)
    recall = tmp / np.minimum(k, X_true_binary.sum(axis=1))
    return recall


def create_qrel(holdout: np.array, binary: bool, user_index: Dict[int, str] = None, exclude: np.array = None) -> Dict[
    str, Dict[str, float]]:
    # holdout -> binary numpy.array of [n_users, n_items]
    # binary -> True if the input is 0/1 (ex. holdout) or False if it's a float/prediction
    # exclude -> which items to exclude in the qrel e.g this can be the input fet to a AE recommender
    # if index is provided, it is a dict that s.t numeric user-id in holdout -> user-id
    # similarly for item_index
    # this makes comparision using actual user ids further down the line straightforward.

    n_users = holdout.shape[0]
    if not user_index:
        user_index = {i: i for i in range(n_users)}

    # qrel for pytrec_eval is
    # query id (here, a user) -> { doc_id_1 (item): relevance_1, ... }
    qrel = {}
    for uid, user in enumerate(holdout):

        if binary:
            _, idx = user.nonzero()
            scores = np.ones_like(idx, dtype=int).tolist()
            user_qrel = {str(i): s for (i, s) in zip(idx, scores)}
        else:
            # sort reverse https://stackoverflow.com/a/16486305
            idx = (-user).argsort()
            scores = user[idx]
            user_qrel = {str(i): float(s) for (i, s) in zip(idx, scores)}

        # exclude scores with -inf

        # exclude these items if provided
        if exclude is not None:
            for item in exclude[uid].nonzero():
                if str(item) in user_qrel:
                    del user_qrel[str(item)]

        qrel[str(user_index[uid])] = user_qrel

    return qrel


class PyTrecEvaluator:
    def __init__(self, metrics: List[str], holdout: np.array, user_index: Dict[int, str] = None,
                 item_index: Dict[int, str] = None) -> None:
        self.metrics = metrics
        self.user_index = user_index
        self.item_index = item_index
        self.gt_holdout = holdout
        self.gt_qrel = create_qrel(self.gt_holdout, binary=True, user_index=self.user_index)
        self.evaluator = pytrec_eval.RelevanceEvaluator(self.gt_qrel, self.metrics)

    def compute_metrics(self, predictions: np.array, exclude: np.array = None) -> Dict[str, Dict[str, float]]:
        if exclude is not None:
            assert predictions.shape == exclude.shape

        pred_qrel = create_qrel(predictions, binary=False, user_index=self.user_index, exclude=exclude)
        # print(pred_qrel['0'])
        return self.evaluator.evaluate(pred_qrel)

    def aggregate(self, results: Dict[str, Dict[str, float]], aggregate: str = "mean_queries") -> Dict[str, Any]:
        if aggregate == "mean":
            # metric -> (mean, std)
            res_temp = self.aggregate(results, "gather")
            final = {}
            for metric, values in res_temp.items():
                final[metric] = (np.mean(values), np.std(values))
            return final

        if aggregate == "gather":
            # metric -> [res_q1, res_q2, ...]
            final = defaultdict(list)
            for qid, qvals in results.items():
                for metric, met_val in qvals.items():
                    final[metric].append(met_val)
            return final

        raise NotImplementedError(f"unknown aggregation {aggregate}")
