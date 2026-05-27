"""Подготовка данных и sparse user-item матрицы для ItemKNN."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
import scipy.sparse as sp

from config import (
    ALPHA_CONTACT,
    ALPHA_VIEW,
    CUTOFF_MS,
    SUBMISSION_K,
    VOCAB_SIZE,
    WEIGHTED,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Vocab:
    item2idx: dict[int, int]
    idx2item: list[int]
    popular_items: list[int]

    @property
    def size(self) -> int:
        return len(self.idx2item) - 1

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "item2idx": {str(k): v for k, v in self.item2idx.items()},
            "idx2item": self.idx2item,
            "popular_items": self.popular_items,
        }
        path.write_text(json.dumps(payload), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "Vocab":
        payload = json.loads(path.read_text(encoding="utf-8"))
        item2idx = {int(k): int(v) for k, v in payload["item2idx"].items()}
        return cls(
            item2idx=item2idx,
            idx2item=[int(x) for x in payload["idx2item"]],
            popular_items=[int(x) for x in payload["popular_items"]],
        )


@dataclass(frozen=True)
class UserIndex:
    user2idx: dict[int, int]
    idx2user: list[int]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "user2idx": {str(k): v for k, v in self.user2idx.items()},
            "idx2user": self.idx2user,
        }
        path.write_text(json.dumps(payload), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "UserIndex":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            user2idx={int(k): int(v) for k, v in payload["user2idx"].items()},
            idx2user=[int(x) for x in payload["idx2user"]],
        )


def eval_user_events_path(data_dir: Path) -> Path:
    for name in ("eval_user_events.pq", "eval_user_events.parquet"):
        p = data_dir / name
        if p.exists():
            return p
    raise FileNotFoundError(f"eval_user_events not found in {data_dir}")


def load_contact_eids(path: Path) -> set[int]:
    df = pl.read_csv(path)
    col = "mapped_eid" if "mapped_eid" in df.columns else df.columns[0]
    return {int(x) for x in df[col].to_list()}


def train_parquet_paths(data_dir: Path) -> list[Path]:
    d = data_dir / "train_data"
    paths = sorted(d.glob("*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No parquet in {d}")
    return paths


def build_item_counts(
    data_dir: Path,
    contact_eids: set[int],
    *,
    use_train: bool = True,
) -> pl.DataFrame:
    ev_path = eval_user_events_path(data_dir)
    contact_list = list(contact_eids)

    ev = (
        pl.scan_parquet(str(ev_path))
        .filter(pl.col("timestamp") < CUTOFF_MS)
        .select(["item_id", "eid"])
    )
    ev_counts = (
        ev.with_columns(
            pl.when(pl.col("eid").is_in(contact_list))
            .then(pl.lit(ALPHA_CONTACT))
            .otherwise(pl.lit(ALPHA_VIEW))
            .alias("w")
        )
        .group_by("item_id")
        .agg(pl.col("w").sum().alias("cnt"))
        .collect(streaming=True)
    )

    if not use_train:
        return ev_counts.sort("cnt", descending=True)

    parts: list[pl.DataFrame] = [ev_counts]
    for path in train_parquet_paths(data_dir):
        chunk = (
            pl.scan_parquet(str(path))
            .filter(pl.col("timestamp") < CUTOFF_MS)
            .select(["item_id", "eid"])
            .with_columns(
                pl.when(pl.col("eid").is_in(contact_list))
                .then(pl.lit(ALPHA_CONTACT))
                .otherwise(pl.lit(ALPHA_VIEW))
                .alias("w")
            )
            .group_by("item_id")
            .agg(pl.col("w").sum().alias("cnt"))
            .collect(streaming=True)
        )
        if not chunk.is_empty():
            parts.append(chunk)

    return (
        pl.concat(parts, how="vertical")
        .group_by("item_id")
        .agg(pl.col("cnt").sum())
        .sort("cnt", descending=True)
    )


def build_vocab(
    data_dir: Path,
    contact_eids: set[int],
    vocab_size: int = VOCAB_SIZE,
    *,
    use_train: bool | None = None,
) -> Vocab:
    if use_train is None:
        from config import USE_TRAIN_FOR_VOCAB

        use_train = USE_TRAIN_FOR_VOCAB
    counts = build_item_counts(data_dir, contact_eids, use_train=use_train)
    top = counts.head(vocab_size)
    idx2item = [0]
    item2idx: dict[int, int] = {0: 0}
    for row in top.iter_rows():
        item_id = int(row[0])
        item2idx[item_id] = len(idx2item)
        idx2item.append(item_id)
    popular = [int(x) for x in top["item_id"].head(SUBMISSION_K * 4).to_list()]
    logger.info("Vocab size=%s (incl. padding)", len(idx2item) - 1)
    return Vocab(item2idx=item2idx, idx2item=idx2item, popular_items=popular)


def iter_eval_users(data_dir: Path) -> pl.DataFrame:
    return pl.read_csv(data_dir / "eval_users.csv").select(
        pl.col("user_id").cast(pl.UInt32).unique()
    )


def build_user_index(eval_users: pl.DataFrame) -> UserIndex:
    uids = [int(x) for x in eval_users["user_id"].to_list()]
    idx2user = [0] + uids
    user2idx = {uid: i for i, uid in enumerate(idx2user)}
    return UserIndex(user2idx=user2idx, idx2user=idx2user)


def build_user_item_matrix(
    data_dir: Path,
    vocab: Vocab,
    user_index: UserIndex,
    contact_eids: set[int],
) -> sp.csr_matrix:
    """Sparse R: (n_users, n_items) from eval_user_events."""
    ev_path = eval_user_events_path(data_dir)
    valid_items = set(vocab.item2idx.keys()) - {0}
    valid_users = set(user_index.user2idx.keys()) - {0}
    contact_list = list(contact_eids)

    lf = (
        pl.scan_parquet(str(ev_path))
        .filter(pl.col("timestamp") < CUTOFF_MS)
        .select(
            [
                pl.col("user_id").cast(pl.UInt32),
                pl.col("item_id").cast(pl.UInt32),
                pl.col("eid"),
            ]
        )
        .filter(pl.col("user_id").is_in(list(valid_users)))
        .filter(pl.col("item_id").is_in(list(valid_items)))
    )

    if WEIGHTED:
        lf = lf.with_columns(
            pl.when(pl.col("eid").is_in(contact_list))
            .then(pl.lit(float(ALPHA_CONTACT)))
            .otherwise(pl.lit(float(ALPHA_VIEW)))
            .alias("val")
        )
        agg = lf.group_by(["user_id", "item_id"]).agg(pl.col("val").max().alias("val"))
    else:
        agg = lf.group_by(["user_id", "item_id"]).len().with_columns(
            pl.lit(1.0).alias("val")
        ).select(["user_id", "item_id", "val"])

    df = agg.collect(streaming=True)
    if df.is_empty():
        n_users = len(user_index.idx2user)
        n_items = len(vocab.idx2item)
        return sp.csr_matrix((n_users, n_items), dtype=np.float32)

    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    for uid, iid, val in df.iter_rows():
        rows.append(user_index.user2idx[int(uid)])
        cols.append(vocab.item2idx[int(iid)])
        vals.append(float(val))

    mat = sp.coo_matrix(
        (vals, (rows, cols)),
        shape=(len(user_index.idx2user), len(vocab.idx2item)),
        dtype=np.float32,
    ).tocsr()
    mat.sum_duplicates()
    logger.info(
        "User-item matrix shape=%s nnz=%s",
        mat.shape,
        mat.nnz,
    )
    return mat


def build_seen_items(data_dir: Path, eval_users: pl.DataFrame) -> dict[int, set[int]]:
    ev_path = eval_user_events_path(data_dir)
    df = (
        pl.scan_parquet(str(ev_path))
        .filter(pl.col("timestamp") < CUTOFF_MS)
        .join(eval_users.lazy(), on="user_id", how="inner")
        .select(["user_id", "item_id"])
        .unique()
        .collect(streaming=True)
    )
    seen: dict[int, set[int]] = {}
    for row in df.iter_rows():
        uid, iid = int(row[0]), int(row[1])
        seen.setdefault(uid, set()).add(iid)
    return seen


def pad_submission(
    rows: list[tuple[int, int]],
    eval_users: pl.DataFrame,
    popular_items: list[int],
    k: int = SUBMISSION_K,
) -> pl.DataFrame:
    k = k or SUBMISSION_K
    by_user: dict[int, list[int]] = {}
    for uid, iid in rows:
        by_user.setdefault(uid, []).append(iid)

    out_rows: list[tuple[int, int]] = []
    for uid in eval_users["user_id"].to_list():
        uid = int(uid)
        taken = set(by_user.get(uid, []))
        items = list(by_user.get(uid, []))
        for iid in popular_items:
            if len(items) >= k:
                break
            if iid not in taken:
                items.append(iid)
                taken.add(iid)
        for iid in items[:k]:
            out_rows.append((uid, iid))

    return pl.DataFrame(out_rows, schema={"user_id": pl.UInt32, "item_id": pl.UInt32}, orient="row")
