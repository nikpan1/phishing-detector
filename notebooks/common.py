"""Wspólne dane i pomiary; bez sieci i bez wykonywania zawartości próbek."""
from pathlib import Path
from time import perf_counter
import gzip
import hashlib
import io
import json
import pickle
import re

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
SEED = 42


def load_splits(limit=8000):
    """Historical comparison notebooks now read an explicitly supplied external file."""
    import os
    import sys
    dataset = os.environ.get('PHISHING_DATASET')
    if not dataset:
        raise ValueError('Ustaw PHISHING_DATASET na lokalny plik CSV, JSON lub JSONL z text,label')
    sys.path.insert(0, str(ROOT / 'phishing-detector' / 'src'))
    from data import load_splits as external_splits
    train, validation, test, info = external_splits(dataset, seed=SEED, limit=limit)
    for split in (train, validation, test):
        split['input'] = split.text
    print(json.dumps(info, indent=2))
    return train, validation, test


def classification_metrics(y, pred, scores):
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return dict(accuracy=accuracy_score(y, pred), precision=precision_score(y, pred, zero_division=0),
                recall=recall_score(y, pred, zero_division=0), f1=f1_score(y, pred, zero_division=0),
                fpr=fp / max(1, fp + tn), roc_auc=roc_auc_score(y, scores),
                average_precision=average_precision_score(y, scores),
                tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp))


def choose_threshold(y, scores):
    """Maksymalizuj F1 wyłącznie na walidacji, nigdy na teście."""
    candidates = np.unique(np.r_[np.quantile(scores, np.linspace(0, 1, 101)),
                                  np.nextafter(max(scores), np.inf)])
    return max(candidates, key=lambda t: f1_score(y, scores >= t, zero_division=0))


def timed_prediction(predict, texts, repeats=3):
    predict(texts[:32])  # rozgrzewka; również mały batch dla HDBSCAN
    durations = []
    for _ in range(repeats):
        start = perf_counter()
        scores = np.asarray(predict(texts))
        durations.append((perf_counter() - start) * 1000 / len(texts))
    return scores, float(np.median(durations))


def size_mib(obj):
    return len(pickle.dumps(obj, protocol=5)) / 1024**2


def save_results(name, rows):
    result = pd.DataFrame(rows)
    target = ROOT / 'results'
    target.mkdir(exist_ok=True)
    result.to_csv(target / f'{name}.csv', index=False)
    return result
