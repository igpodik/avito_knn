#!/usr/bin/env python3
"""Построение ItemKNN item×item similarity и сохранение артефактов."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import scipy.sparse as sp

from config import (
    ARTIFACTS_DIR,
    BLOCK_SIZE,
    DATA_DIR,
    POLARS_MAX_THREADS,
    SHRINK,
    TOPK,
    TRAIN_MAX_USERS,
    VOCAB_SIZE,
    WEIGHTED,
    WEIGHTING,
)
from data import (
    build_user_index,
    build_user_item_matrix,
    build_vocab,
    iter_eval_users,
    load_contact_eids,
)
from model import ItemKNN


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, default=DATA_DIR)
    p.add_argument("--artifacts-dir", type=Path, default=ARTIFACTS_DIR)
    p.add_argument("--vocab-size", type=int, default=VOCAB_SIZE)
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    os.environ["POLARS_MAX_THREADS"] = str(POLARS_MAX_THREADS)

    data_dir = args.data_dir.expanduser().resolve()
    artifacts_dir = args.artifacts_dir.expanduser().resolve()
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    contact = load_contact_eids(data_dir / "contact_eids.csv")
    logging.info("Building vocabulary (top %s items)…", args.vocab_size)
    vocab = build_vocab(data_dir, contact, vocab_size=args.vocab_size)
    vocab.save(artifacts_dir / "vocab.json")

    eval_users = iter_eval_users(data_dir)
    if TRAIN_MAX_USERS > 0:
        eval_users = eval_users.head(TRAIN_MAX_USERS)
        logging.info("Limited to %s users", eval_users.height)

    user_index = build_user_index(eval_users)
    user_index.save(artifacts_dir / "user2idx.json")

    logging.info("Building user-item matrix…")
    adj_mat = build_user_item_matrix(data_dir, vocab, user_index, contact)
    sp.save_npz(artifacts_dir / "interaction.npz", adj_mat)

    logging.info("Fitting ItemKNN (weighting=%s, topk=%s)…", WEIGHTING, TOPK)
    model = ItemKNN(
        topk=TOPK,
        shrink=SHRINK,
        feature_weighting=WEIGHTING,
        block_size=BLOCK_SIZE,
    )
    model.fit(adj_mat)
    model.save(artifacts_dir)

    meta = {
        "vocab_size": vocab.size,
        "n_users": len(user_index.idx2user),
        "matrix_shape": list(adj_mat.shape),
        "matrix_nnz": int(adj_mat.nnz),
        "w_shape": list(model.artifacts.w_sparse.shape),
        "w_nnz": int(model.artifacts.w_sparse.nnz),
        "topk": TOPK,
        "shrink": SHRINK,
        "weighting": WEIGHTING,
        "block_size": BLOCK_SIZE,
        "weighted": WEIGHTED,
    }
    (artifacts_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    logging.info("Saved artifacts → %s", artifacts_dir)


if __name__ == "__main__":
    main(sys.argv[1:])
