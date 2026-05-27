"""
ItemKNN: item-based collaborative filtering with cosine similarity.

Jun Wang et al., Unifying user-based and item-based collaborative filtering
approaches by similarity fusion. SIGIR 2006.
Ported from yoongi0428/RecSys_PyTorch (scipy-only, no PyTorch).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class ItemKNNArtifacts:
    w_sparse: sp.csr_matrix


class ItemKNN:
    """Item-based KNN with optional BM25/TF-IDF weighting."""

    def __init__(
        self,
        topk: int = 100,
        shrink: float = 100.0,
        feature_weighting: str = "bm25",
        block_size: int = 500,
    ) -> None:
        self.topk = topk
        self.shrink = shrink
        self.feature_weighting = feature_weighting.lower()
        self.block_size = block_size
        if self.feature_weighting not in ("tf-idf", "bm25", "none"):
            raise ValueError(f"Unknown feature_weighting: {feature_weighting}")
        self._artifacts: ItemKNNArtifacts | None = None

    @property
    def artifacts(self) -> ItemKNNArtifacts:
        if self._artifacts is None:
            raise RuntimeError("ItemKNN not fitted")
        return self._artifacts

    @staticmethod
    def okapi_bm25(rating_matrix: sp.csr_matrix, k1: float = 1.2, b: float = 0.75) -> sp.csr_matrix:
        assert 0 < b < 1, "okapi_BM25: B must be in (0,1)"
        assert k1 > 0, "okapi_BM25: K1 must be > 0"

        rating_matrix = sp.coo_matrix(rating_matrix)
        n = float(rating_matrix.shape[0])
        idf = np.log(n / (1 + np.bincount(rating_matrix.col, minlength=rating_matrix.shape[1])))
        row_sums = np.ravel(rating_matrix.sum(axis=1))
        average_length = row_sums.mean()
        length_norm = (1.0 - b) + b * row_sums / average_length
        data = (
            rating_matrix.data
            * (k1 + 1.0)
            / (k1 * length_norm[rating_matrix.row] + rating_matrix.data)
            * idf[rating_matrix.col]
        )
        return sp.csr_matrix(
            (data, (rating_matrix.row, rating_matrix.col)),
            shape=rating_matrix.shape,
            dtype=np.float32,
        )

    @staticmethod
    def tf_idf(rating_matrix: sp.csr_matrix) -> sp.csr_matrix:
        rating_matrix = sp.coo_matrix(rating_matrix)
        n = float(rating_matrix.shape[0])
        idf = np.log(n / (1 + np.bincount(rating_matrix.col, minlength=rating_matrix.shape[1])))
        data = np.sqrt(rating_matrix.data) * idf[rating_matrix.col]
        return sp.csr_matrix(
            (data.astype(np.float32), (rating_matrix.row, rating_matrix.col)),
            shape=rating_matrix.shape,
            dtype=np.float32,
        )

    def _apply_weighting(self, train_matrix: sp.csr_matrix) -> sp.csr_matrix:
        item_matrix = train_matrix.T.tocsr()
        if self.feature_weighting == "tf-idf":
            return self.tf_idf(item_matrix).T.tocsr()
        if self.feature_weighting == "bm25":
            return self.okapi_bm25(item_matrix).T.tocsr()
        return train_matrix.tocsr().astype(np.float32)

    def fit(self, train_matrix: sp.csr_matrix) -> None:
        """Build sparse item×item similarity matrix W."""
        train_matrix = self._apply_weighting(train_matrix)
        train_matrix = train_matrix.tocsc()
        num_items = train_matrix.shape[1]
        block_size = self.block_size

        sum_of_squared = np.array(train_matrix.power(2).sum(axis=0)).ravel()
        sum_of_squared = np.sqrt(sum_of_squared)

        values: list[float] = []
        rows: list[int] = []
        cols: list[int] = []

        blocks = range(0, num_items, block_size)
        for start_col_block in tqdm(blocks, desc="ItemKNN fit", unit="block"):
            end_col_block = min(start_col_block + block_size, num_items)
            this_block_size = end_col_block - start_col_block

            item_data = train_matrix[:, start_col_block:end_col_block].toarray()
            if item_data.ndim == 1:
                item_data = item_data.reshape(-1, 1)

            this_block_weights = train_matrix.T.dot(item_data)

            for col_index_in_block in range(this_block_size):
                this_column_weights = this_block_weights[:, col_index_in_block]

                column_index = col_index_in_block + start_col_block
                this_column_weights[column_index] = 0.0

                denominator = sum_of_squared[column_index] * sum_of_squared + self.shrink + 1e-6
                this_column_weights = np.multiply(this_column_weights, 1.0 / denominator)

                k = min(self.topk, num_items)
                relevant_items_partition = (-this_column_weights).argpartition(k - 1)[:k]
                relevant_items_partition_sorting = np.argsort(
                    -this_column_weights[relevant_items_partition]
                )
                top_k_idx = relevant_items_partition[relevant_items_partition_sorting]

                not_zeros_mask = this_column_weights[top_k_idx] != 0.0
                num_not_zeros = int(np.sum(not_zeros_mask))

                values.extend(this_column_weights[top_k_idx][not_zeros_mask])
                rows.extend(top_k_idx[not_zeros_mask])
                cols.extend(np.ones(num_not_zeros, dtype=np.int64) * column_index)

        w_sparse = sp.csr_matrix(
            (values, (rows, cols)),
            shape=(num_items, num_items),
            dtype=np.float32,
        )
        self._artifacts = ItemKNNArtifacts(w_sparse=w_sparse)
        logger.info(
            "ItemKNN fit done W shape=%s nnz=%s",
            w_sparse.shape,
            w_sparse.nnz,
        )

    def score_batch(self, user_rows: sp.csr_matrix) -> np.ndarray:
        """Scores for batch of users: (batch, n_items)."""
        user_rows = user_rows.tocsr().astype(np.float32)
        preds = user_rows @ self.artifacts.w_sparse
        if sp.issparse(preds):
            return preds.toarray().astype(np.float32, copy=False)
        return np.asarray(preds, dtype=np.float32)

    def save(self, artifacts_dir: Path | str) -> None:
        path = Path(artifacts_dir)
        path.mkdir(parents=True, exist_ok=True)
        sp.save_npz(path / "item_knn.npz", self.artifacts.w_sparse)

    @classmethod
    def load(cls, artifacts_dir: Path | str) -> "ItemKNN":
        path = Path(artifacts_dir)
        model = cls()
        w_sparse = sp.load_npz(path / "item_knn.npz").tocsr()
        model._artifacts = ItemKNNArtifacts(w_sparse=w_sparse)
        return model
