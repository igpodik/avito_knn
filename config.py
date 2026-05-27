"""Константы ItemKNN пайплайна для Avito ML Cup."""

from __future__ import annotations

import datetime as dt
import os
from pathlib import Path

# --- Время (согласовано с baseline: cutoff 2026-04-15 UTC) ---
CUTOFF_UTC = dt.datetime(2026, 4, 8, 0, 0, 0, tzinfo=dt.timezone.utc)
CUTOFF_MS = int(CUTOFF_UTC.timestamp() * 1000)

# --- Submission ---
SUBMISSION_K = 200
RANDOM_SEED = 42

# --- ItemKNN гиперпараметры ---
VOCAB_SIZE = int(os.environ.get("KNN_VOCAB", "200000"))
TOPK = int(os.environ.get("KNN_TOPK", "100"))
SHRINK = float(os.environ.get("KNN_SHRINK", "100"))
WEIGHTING = os.environ.get("KNN_WEIGHTING", "bm25").lower()
BLOCK_SIZE = int(os.environ.get("KNN_BLOCK_SIZE", "500"))
WEIGHTED = os.environ.get("KNN_WEIGHTED", "1") == "1"

PRED_BATCH = int(os.environ.get("KNN_PRED_BATCH", "512"))
TRAIN_MAX_USERS = int(os.environ.get("KNN_TRAIN_MAX_USERS", "0"))
PRED_MAX_USERS = int(os.environ.get("KNN_PRED_MAX_USERS", "0"))
USE_TRAIN_FOR_VOCAB = os.environ.get("KNN_USE_TRAIN", "1") == "1"

# --- Данные ---
ALPHA_CONTACT = 5.0
ALPHA_VIEW = 1.0

DATA_DIR = Path(os.environ.get("DATA_DIR", str(Path(__file__).resolve().parent.parent.parent / "data")))
ARTIFACTS_DIR = Path(os.environ.get("ARTIFACTS_DIR", str(Path(__file__).resolve().parent / "artifacts")))

POLARS_MAX_THREADS = int(os.environ.get("POLARS_MAX_THREADS", "8"))
