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
    """Sprawdź cały zbiór, usuń duplikaty, pobierz próbę, podziel 60/20/20."""
    directory = ROOT / 'data'
    manifest = json.loads((directory / 'manifest.json').read_text())
    compressed = io.BytesIO()
    for part in manifest['parts']:
        raw = (directory / part['file']).read_bytes()
        assert len(raw) == part['bytes']
        assert hashlib.sha256(raw).hexdigest() == part['sha256'], part['file']
        compressed.write(raw)
    compressed.seek(0)
    raw = gzip.decompress(compressed.read())
    assert hashlib.sha256(raw).hexdigest() == manifest['original_sha256']
    records = json.loads(raw)
    assert len(records) == manifest['records']
    frame = pd.DataFrame(records)[['text', 'label']]
    assert frame.text.map(lambda s: isinstance(s, str)).all()
    assert set(frame.label.unique()) == {0, 1}
    # Ten sam klucz do deduplikacji i sprawdzenia rozłączności podziałów.
    frame['key'] = frame.text.map(
        lambda s: hashlib.sha256(re.sub(r'\s+', ' ', s[:4000]).strip().lower().encode()).hexdigest())
    conflicting = frame.groupby('key').label.nunique()
    conflicts = set(conflicting[conflicting > 1].index)
    valid = frame.text.str.strip().ne('') & ~frame.key.isin(conflicts)
    frame = frame[valid].drop_duplicates('key').reset_index(drop=True)
    if limit is not None and len(frame) > limit:
        frame, _ = train_test_split(frame, train_size=limit, stratify=frame.label,
                                    random_state=SEED)
    train, rest = train_test_split(frame, test_size=0.4, stratify=frame.label,
                                   random_state=SEED)
    validation, test = train_test_split(rest, test_size=0.5, stratify=rest.label,
                                       random_state=SEED)
    for left, right in [(train, validation), (train, test), (validation, test)]:
        assert set(left.key).isdisjoint(right.key)
    # Wspólny limit wejścia: czas i RAM nie zależą od wyjątkowo długiego HTML.
    for split in (train, validation, test):
        split['input'] = split.text.str.slice(0, 4000)
    info = dict(source_records=manifest['records'], deduplicated_sample=len(frame),
                conflicting_keys_removed=len(conflicts), seed=SEED,
                train=len(train), validation=len(validation), test=len(test),
                max_input_chars=4000,
                split_sha256={name: hashlib.sha256('\n'.join(sorted(s.key)).encode()).hexdigest()
                              for name, s in [('train',train),('validation',validation),('test',test)]})
    print(json.dumps(info, indent=2))
    return train.reset_index(drop=True), validation.reset_index(drop=True), test.reset_index(drop=True)


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
