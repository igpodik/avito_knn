#!/usr/bin/env python3
"""Инференс ItemKNN → submission.csv (160 items / user)."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import polars as pl
import scipy.sparse as sp

from config import (
    ARTIFACTS_DIR,
    DATA_DIR,
    PRED_BATCH,
    PRED_MAX_USERS,
    POLARS_MAX_THREADS,
    SUBMISSION_K,
)
from data import (
    UserIndex,
    Vocab,
    build_seen_items,
    build_user_index,
    build_user_item_matrix,
    iter_eval_users,
    load_contact_eids,
    pad_submission,
)
from model import ItemKNN


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, default=DATA_DIR)
    p.add_argument("--artifacts-dir", type=Path, default=ARTIFACTS_DIR)
    p.add_argument("--out", type=Path, default=Path("submission.csv"))
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def _top_k_items(
    scores: np.ndarray,
    k: int,
    seen: set[int],
    idx2item: list[int],
) -> list[int]:
    n = scores.shape[0]
    if n <= k + len(seen):
        order = np.argsort(-scores)
    else:
        part = np.argpartition(-scores, k + len(seen))[: k + len(seen)]
        order = part[np.argsort(-scores[part])]
    picked: list[int] = []
    for j in order:
        if j == 0:
            continue
        raw = idx2item[j]
        if raw in seen:
            continue
        picked.append(raw)
        if len(picked) >= k:
            break
    return picked


def score_users(
    model: ItemKNN,
    adj_mat: sp.csr_matrix,
    user_index: UserIndex,
    vocab: Vocab,
    user_ids: list[int],
    seen: dict[int, set[int]],
    k: int = SUBMISSION_K,
) -> list[tuple[int, int]]:
    rows: list[tuple[int, int]] = []
    idx2item = vocab.idx2item

    for start in range(0, len(user_ids), PRED_BATCH):
        batch_uids = user_ids[start : start + PRED_BATCH]
        batch_rows: list[int] = []
        for uid in batch_uids:
            idx = user_index.user2idx.get(uid)
            if idx is None:
                batch_rows.append(-1)
            else:
                batch_rows.append(idx)

        valid_pairs = [(i, r) for i, r in enumerate(batch_rows) if r >= 0]
        if not valid_pairs:
            continue

        mat_rows = [r for _, r in valid_pairs]
        user_block = adj_mat[mat_rows]
        scores = model.score_batch(user_block)

        for local_i, (batch_i, _) in enumerate(valid_pairs):
            uid = batch_uids[batch_i]
            picked = _top_k_items(
                scores[local_i],
                k,
                seen.get(uid, set()),
                idx2item,
            )
            rows.extend((uid, iid) for iid in picked)

    return rows


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    os.environ["POLARS_MAX_THREADS"] = str(POLARS_MAX_THREADS)

    data_dir = args.data_dir.expanduser().resolve()
    artifacts_dir = args.artifacts_dir.expanduser().resolve()
    out_csv = args.out.expanduser().resolve()

    vocab = Vocab.load(artifacts_dir / "vocab.json")
    user_index = UserIndex.load(artifacts_dir / "user2idx.json")
    model = ItemKNN.load(artifacts_dir)

    eval_users = iter_eval_users(data_dir)
    if PRED_MAX_USERS > 0:
        eval_users = eval_users.head(PRED_MAX_USERS)
        logging.info("Limited predict to %s users", eval_users.height)

    user_ids = [int(x) for x in eval_users["user_id"].to_list()]

    interaction_path = artifacts_dir / "interaction.npz"
    if interaction_path.exists() and PRED_MAX_USERS == 0:
        logging.info("Loading cached interaction matrix")
        adj_mat = sp.load_npz(interaction_path).tocsr()
    else:
        logging.info("Rebuilding interaction matrix for predict users")
        pred_index = build_user_index(eval_users)
        contact = load_contact_eids(data_dir / "contact_eids.csv")
        adj_mat = build_user_item_matrix(data_dir, vocab, pred_index, contact)
        user_index = pred_index

    logging.info("Building seen-item sets…")
    seen = build_seen_items(data_dir, eval_users)

    logging.info("Scoring %s users…", len(user_ids))
    rows = score_users(model, adj_mat, user_index, vocab, user_ids, seen, k=SUBMISSION_K)

    out = pad_submission(rows, eval_users, vocab.popular_items, k=SUBMISSION_K)
    out.write_csv(out_csv)
    logging.info(
        "Wrote %s rows, %s users → %s",
        out.height,
        out["user_id"].n_unique(),
        out_csv,
    )


if __name__ == "__main__":
    main(sys.argv[1:])
