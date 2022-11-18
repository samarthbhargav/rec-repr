import time
import os

from src import utils
from run import get_model, evaluate
import torch
from src.data import DataMgr
from run import generate
import logging
import argparse
import numpy as np

log = logging.getLogger(__name__)


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument("--run", type=str, required=True, help="location of output file from run.py")
    parser.add_argument("--result_loc", type=str, required=True, help="location to store test output")

    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--batch_size', type=int, default=500)
    return parser


def fix_paths(args, run_args):
    """
    Fix paths. This may occur due to transferring between systems where the root path may differ
    :param args:
    :param run_args:
    :return:
    """
    ref_path = os.path.split(args.run)[0]
    for path_arg in ["out", "save", "model_config"]:
        if not os.path.exists(run_args[path_arg]):
            log.info(f"path {run_args[path_arg]} doesn't exist. attempting to fix")
            curr = run_args[path_arg]
            fold, fil = os.path.split(curr)
            if os.path.exists(os.path.join(ref_path, fil)):
                run_args[path_arg] = os.path.join(ref_path, fil)
                log.info(f"\t fixed to {run_args[path_arg]}")
            else:
                raise ValueError(f"path {run_args[path_arg]} doesn't exist")


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()

    utils.configure_logging("test", False)
    utils.set_seed(args.seed)

    log.info(f"args: {vars(args)}")
    device = torch.device(args.device)

    run_out = utils.read_json(args.run, zipped=True)
    model_config = run_out["model_config"]
    run_args = run_out["args"]
    fix_paths(args, run_args)
    log.info(f"run args: {run_args}")
    log.info(f"run model_config: {model_config}")

    data_mgr = DataMgr(run_args["dataset"])
    train_data, val_in_data, val_out_data, test_in_data, test_out_data = data_mgr.get_data()
    n_items = train_data.shape[1]
    model = get_model(model_config)
    model = model.to(device)
    model.load_state_dict(torch.load(run_args["save"], map_location=device))

    log.info(f"Evaluating test set performance")

    test_metrics = ["ndcg_cut_100", "recall_50"]
    test_scores = evaluate(model,
                           data_in=test_in_data,
                           data_out=test_out_data,
                           metrics=test_metrics,
                           device=device,
                           samples_perc_per_epoch=1.,
                           batch_size=args.batch_size,
                           fast=False)
    log.info(f"Test set evaluation complete:")
    for met, (mean, std) in test_scores.items():
        log.info(f"\t{met: >15}: {mean:0.4f} ({std:.4f})")

    out = {
        "test_scores": test_scores,
        "args": vars(args)
    }

    utils.write_json(out, args.result_loc, zipped=args.result_loc.endswith(".zip"))
    log.info(f"Wrote output to {args.result_loc}")
