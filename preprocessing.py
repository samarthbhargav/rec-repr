# based on https://github.com/dawenl/vae_cf
import gzip
import json
import os

import sys

import numpy as np

import pandas as pd

import argparse

from src import utils


def get_count(tp, id):
    playcount_groupbyid = tp[[id]].groupby(id, as_index=False)
    count = playcount_groupbyid.size()
    return count


def filter_triplets(tp, min_uc, min_sc):
    if min_sc > 0:
        itemcount = get_count(tp, 'movieId')

        tp = tp[tp['movieId'].isin(itemcount.index[itemcount >= min_sc])]

    if min_uc > 0:
        usercount = get_count(tp, 'userId')
        tp = tp[tp['userId'].isin(usercount.index[usercount >= min_uc])]

    usercount, itemcount = get_count(tp, 'userId'), get_count(tp, 'movieId')
    return tp, usercount, itemcount


def split_train_test_proportion(data, test_prop=0.2):
    data_grouped_by_user = data.groupby('userId')
    tr_list, te_list = list(), list()

    np.random.seed(98765)

    for i, (_, group) in enumerate(data_grouped_by_user):
        n_items_u = len(group)

        if n_items_u >= 5:
            idx = np.zeros(n_items_u, dtype='bool')
            idx[np.random.choice(n_items_u, size=int(test_prop * n_items_u), replace=False).astype('int64')] = True

            tr_list.append(group[np.logical_not(idx)])
            te_list.append(group[idx])
        else:
            tr_list.append(group)

        if i % 1000 == 0:
            print("%d users sampled" % i)
            sys.stdout.flush()

    data_tr = pd.concat(tr_list)
    data_te = pd.concat(te_list)

    return data_tr, data_te


def numerize(tp, profile2id, show2id):
    uid = list(map(lambda x: profile2id[x], tp['userId']))
    sid = list(map(lambda x: show2id[x], tp['movieId']))
    return pd.DataFrame(data={'uid': uid, 'sid': sid}, columns=['uid', 'sid'])


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, required=True,
                        choices=["ml-1m", "ml-20m", "msd", "gr", "gr-comics", "gr-children"])
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--threshold', type=float, required=True)
    parser.add_argument('--heldout_users', type=int, required=True)
    parser.add_argument('--min_items_per_user', type=int, default=5)
    parser.add_argument('--min_users_per_item', type=int, default=0)
    return parser


if __name__ == '__main__':

    args = get_parser().parse_args()

    data_folder = args.dataset
    output_dir = args.output_dir
    threshold = args.threshold
    min_uc = args.min_items_per_user
    min_sc = args.min_users_per_item
    n_heldout_users = args.heldout_users

    if not os.path.exists(os.path.join(output_dir, 'test_te.csv')):
        if args.dataset_name == "ml-20m":
            raw_data = pd.read_csv(os.path.join(data_folder, "ratings.csv"), header=0)
        elif args.dataset_name == "ml-1m":
            raw_data = pd.read_csv(os.path.join(data_folder, 'ratings.dat'),
                                   names="userId|movieId|rating|timestamp".split(
                                       "|"),
                                   delimiter="::")
        elif args.dataset_name == "msd":
            raw_data = pd.read_csv(os.path.join(data_folder, "train_triplets.txt"),
                                   names=("userId", "movieId", "rating"),
                                   delimiter="\t")
            # play count -> rating set to 5
            raw_data["rating"] = 5
        elif args.dataset_name.startswith("gr"):
            ratings_file = os.path.join(args.dataset, "ratings.csv")
            if not os.path.exists(ratings_file):
                ratings = []
                interaction_file = {
                    "gr-comics": "goodreads_interactions_comics_graphic.json.gz",
                    "gr-children": "goodreads_interactions_children.json.gz",
                    "gr": "goodreads_reviews_dedup.json.gz"
                }[args.dataset_name]
                with gzip.open(os.path.join(args.dataset, interaction_file), "r") as reader:
                    for line in reader:
                        j = json.loads(line)
                        ratings.append({
                            "userId": j["user_id"],
                            "movieId": j["book_id"],
                            "rating": j["rating"],
                            "timestamp": j["date_added"]
                        })

                print(f"loaded {len(ratings)} ratings. saving to {ratings_file}")
                raw_data = pd.DataFrame(ratings)
                raw_data.to_csv(ratings_file, index=False)
            else:
                raw_data = pd.read_csv(ratings_file)
        else:
            raise ValueError(args.dataset_name)

        raw_data = raw_data[raw_data['rating'] > threshold]

        raw_data, user_activity, item_popularity = filter_triplets(raw_data, min_sc=min_sc, min_uc=min_uc)

        sparsity = 1. * raw_data.shape[0] / (user_activity.shape[0] * item_popularity.shape[0])

        print("After filtering, there are %d watching events from %d users and %d movies (sparsity: %.3f%%)" %
              (raw_data.shape[0], user_activity.shape[0], item_popularity.shape[0], sparsity * 100))

        unique_uid = user_activity.index

        np.random.seed(98765)
        idx_perm = np.random.permutation(unique_uid.size)
        unique_uid = unique_uid[idx_perm]

        n_users = unique_uid.size

        tr_users = unique_uid[:(n_users - n_heldout_users * 2)]
        vd_users = unique_uid[(n_users - n_heldout_users * 2): (n_users - n_heldout_users)]
        te_users = unique_uid[(n_users - n_heldout_users):]

        train_plays = raw_data.loc[raw_data['userId'].isin(tr_users)]

        unique_sid = pd.unique(train_plays['movieId'])

        show2id = dict((sid, i) for (i, sid) in enumerate(unique_sid))
        profile2id = dict((pid, i) for (i, pid) in enumerate(unique_uid))

        os.makedirs(output_dir, exist_ok=True)

        utils.save_pickle(os.path.join(output_dir, "show2id.pkl"), show2id)
        utils.save_pickle(os.path.join(output_dir, "profile2id.pkl"), profile2id)

        with open(os.path.join(output_dir, 'unique_sid.txt'), 'w') as f:
            for sid in unique_sid:
                f.write('%s\n' % sid)

        with open(os.path.join(output_dir, 'unique_uid.txt'), 'w') as f:
            for uid in unique_uid:
                f.write('%s\n' % uid)

        vad_plays = raw_data.loc[raw_data['userId'].isin(vd_users)]
        vad_plays = vad_plays.loc[vad_plays['movieId'].isin(unique_sid)]

        vad_plays_tr, vad_plays_te = split_train_test_proportion(vad_plays)

        test_plays = raw_data.loc[raw_data['userId'].isin(te_users)]
        test_plays = test_plays.loc[test_plays['movieId'].isin(unique_sid)]

        test_plays_tr, test_plays_te = split_train_test_proportion(test_plays)

        train_data = numerize(train_plays, profile2id, show2id)
        train_data.to_csv(os.path.join(output_dir, 'train.csv'), index=False)

        vad_data_tr = numerize(vad_plays_tr, profile2id, show2id)
        vad_data_tr.to_csv(os.path.join(output_dir, 'validation_tr.csv'), index=False)

        vad_data_te = numerize(vad_plays_te, profile2id, show2id)
        vad_data_te.to_csv(os.path.join(output_dir, 'validation_te.csv'), index=False)

        test_data_tr = numerize(test_plays_tr, profile2id, show2id)
        test_data_tr.to_csv(os.path.join(output_dir, 'test_tr.csv'), index=False)

        test_data_te = numerize(test_plays_te, profile2id, show2id)
        test_data_te.to_csv(os.path.join(output_dir, 'test_te.csv'), index=False)