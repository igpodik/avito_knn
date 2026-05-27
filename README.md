# ItemKNN — Item-based Collaborative Filtering для Avito ML Cup

Item-based KNN с cosine similarity и BM25/TF-IDF weighting на базе [RecSys_PyTorch](https://github.com/yoongi0428/RecSys_PyTorch).  
Paper: [Jun Wang et al., SIGIR 2006](http://web4.cs.ucl.ac.uk/staff/jun.wang/papers/2006-sigir06-unifycf.pdf).

Модель **не использует backprop** — `train.py` строит sparse item×item similarity matrix; scoring — `r_u @ W_sparse`.

> **Disclaimer (нейро-трек):** в [`tz.txt`](tz.txt) указано требование «end2end neural network». ItemKNN — классический CF без обучения весов. Для строгого нейро-трека используйте SASRec/TiSASRec.

---

## Алгоритм

```
# train: blocked cosine similarity between item vectors (users as features)
W_sparse[i, j] = cos(i, j) with shrinkage, top-K neighbors per item

# predict:
scores_u = r_u @ W_sparse
```

Optional BM25/TF-IDF weighting на item-строках матрицы `R^T`.

---

## Структура

```
Itemknn/
├── config.py
├── data.py         # vocab, sparse user×item matrix
├── model.py        # ItemKNN fit + score
├── train.py        # build artifacts
├── predict.py      # submission.csv
├── requirements.txt
├── Dockerfile
└── artifacts/
    ├── vocab.json
    ├── user2idx.json
    ├── interaction.npz
    ├── item_knn.npz
    └── meta.json
```

---

## Запуск

```powershell
cd src/experiments/Itemknn
python -m venv .venv
.\.venv\Scripts\pip install -r requirements.txt

.\.venv\Scripts\python.exe train.py
.\.venv\Scripts\python.exe predict.py --out submission.csv
```

---

## Docker

```bash
docker build -t itemknn-avito .
docker run --rm -v /path/to/data:/data -v /path/to/artifacts:/artifacts itemknn-avito python train.py
docker run --rm -v /path/to/data:/data -v /path/to/artifacts:/artifacts itemknn-avito python predict.py --out /artifacts/submission.csv
```

---

## Переменные окружения

| Переменная | Default | Описание |
|------------|---------|----------|
| `KNN_VOCAB` | 200000 | Top-N items |
| `KNN_TOPK` | 100 | Top-K neighbors per item |
| `KNN_SHRINK` | 100 | Shrinkage for cosine |
| `KNN_WEIGHTING` | bm25 | `bm25` / `tf-idf` / `none` |
| `KNN_BLOCK_SIZE` | 500 | Block size for fit |
| `KNN_WEIGHTED` | 1 | Weighted R (contact×5, view×1) |
| `KNN_USE_TRAIN` | 0 | Include train_data in vocab |
| `KNN_TRAIN_MAX_USERS` | 0 | Limit users (0=all) |
| `KNN_PRED_MAX_USERS` | 0 | Limit predict users |

### Smoke-test

```powershell
$env:KNN_VOCAB="10000"
$env:KNN_TRAIN_MAX_USERS="500"
$env:KNN_PRED_MAX_USERS="200"
.\.venv\Scripts\python.exe train.py
.\.venv\Scripts\python.exe predict.py
```

---

## Ссылки

- Paper: [SIGIR 2006](http://web4.cs.ucl.ac.uk/staff/jun.wang/papers/2006-sigir06-unifycf.pdf)
- Code: [RecSys_PyTorch ItemKNN](https://github.com/yoongi0428/RecSys_PyTorch/blob/master/models/ItemKNN.py)
