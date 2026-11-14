"""Local inference, input validation, threshold selection and metrics."""
import csv
import hashlib
import json
import os
from pathlib import Path
import re
from math import sqrt
from time import perf_counter

for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(name, '2')

import joblib
import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score, roc_curve)

HERE = Path(__file__).resolve().parent
DEFAULT_MODEL = HERE / 'models' / 'pipeline.joblib'


def normalize(text, max_chars=4000):
    if not isinstance(text, str) or not text.strip():
        raise ValueError('text musi być niepustym napisem')
    value = re.sub(r'\s+', ' ', text[:max_chars]).strip().lower()[:max_chars].rstrip()
    if not value:
        raise ValueError('brak tekstu w zakresie max_chars')
    return value


def text_key(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n',
                    encoding='utf-8')


def read_records(path, labels=False):
    """JSONL/CSV: one record per line, required text, optional id and label."""
    path = Path(path)
    with path.open(encoding='utf-8-sig', newline='') as handle:
        is_csv = path.suffix.lower() == '.csv'
        rows = csv.DictReader(handle) if is_csv else (json.loads(line) for line in handle if line.strip())
        for index, row in enumerate(rows, 1):
            if not isinstance(row, dict):
                raise ValueError(f'rekord {index}: oczekiwano obiektu')
            if not isinstance(row.get('text'), str) or not row['text'].strip():
                raise ValueError(f'rekord {index}: text musi być niepustym napisem')
            record = {'text': row['text'], 'id': row.get('id', index)}
            if labels:
                label = row.get('label')
                if is_csv and label in ('0', '1'):
                    label = int(label)
                if type(label) is not int or label not in (0, 1):
                    raise ValueError(f'rekord {index}: label musi być 0 lub 1')
                record['label'] = label
            yield record


def load_model(path=DEFAULT_MODEL):
    # joblib is pickle-based: load only trusted model artifacts.
    bundle = joblib.load(path)
    if not isinstance(bundle, dict) or bundle.get('format_version') != 1:
        raise ValueError('nieobsługiwany format modelu')
    if not np.isfinite(bundle['threshold']) or bundle['max_chars'] <= 0:
        raise ValueError('nieprawidłowe parametry modelu')
    if list(bundle['pipeline'].classes_) != [0, 1]:
        raise ValueError('model musi mieć klasy 0 i 1')
    return bundle


def score_texts(bundle, texts):
    normalized = [normalize(text, bundle['max_chars']) for text in texts]
    if not normalized:
        return np.empty(0)
    return np.asarray(bundle['pipeline'].decision_function(normalized))


def score_batches(bundle, texts, batch_size=256):
    if not texts:
        raise ValueError('pusty zbiór danych')
    return np.concatenate([score_texts(bundle, texts[i:i + batch_size])
                           for i in range(0, len(texts), batch_size)])


def select_threshold(y, scores, max_fpr=0.05):
    """Maximize recall on validation subject to its FPR budget; never use test."""
    if not 0 <= max_fpr <= 1 or set(np.unique(y)) != {0, 1}:
        raise ValueError('max_fpr musi należeć do [0,1]; walidacja wymaga obu klas')
    if not np.isfinite(scores).all():
        raise ValueError('wyniki modelu muszą być skończone')
    fpr, recall, thresholds = roc_curve(y, scores, drop_intermediate=False)
    feasible = np.flatnonzero(fpr <= max_fpr + 1e-12)
    chosen = max(feasible, key=lambda i: (recall[i], -fpr[i], thresholds[i]))
    threshold = thresholds[chosen]
    if not np.isfinite(threshold):
        threshold = np.nextafter(np.max(scores), np.inf)
    return float(threshold)


def select_threshold_for_recall(y, scores, min_recall=0.928):
    """Minimize validation FPR at a predefined recall floor, including score ties."""
    if not 0 < min_recall <= 1 or set(np.unique(y)) != {0, 1}:
        raise ValueError('min_recall musi należeć do (0,1]; walidacja wymaga obu klas')
    if not np.isfinite(scores).all():
        raise ValueError('wyniki modelu muszą być skończone')
    fpr, recall, thresholds = roc_curve(y, scores, drop_intermediate=False)
    eligible = np.flatnonzero(recall >= min_recall - 1e-12)
    chosen = min(eligible, key=lambda i: (fpr[i], -recall[i], -thresholds[i]))
    return float(thresholds[chosen])


def wilson_interval(successes, trials):
    """95% Wilson interval; conditional on the supplied evaluation population."""
    if not trials:
        return None
    z = 1.959963984540054
    p = successes / trials
    denominator = 1 + z*z / trials
    center = (p + z*z / (2*trials)) / denominator
    half = z * sqrt(p*(1-p)/trials + z*z/(4*trials*trials)) / denominator
    return [max(0., center-half), min(1., center+half)]


def metrics(y, scores, threshold):
    y = np.asarray(y)
    predictions = np.asarray(scores) >= threshold
    tn, fp, fn, tp = (int(x) for x in confusion_matrix(y, predictions, labels=[0, 1]).ravel())
    both = set(np.unique(y)) == {0, 1}
    return dict(samples=len(y), phishing_samples=int((y == 1).sum()),
                benign_samples=int((y == 0).sum()),
                accuracy=float(accuracy_score(y, predictions)),
                precision=float(precision_score(y, predictions, zero_division=0)),
                recall=float(recall_score(y, predictions, zero_division=0)) if tp + fn else None,
                f1=float(f1_score(y, predictions, zero_division=0)),
                fpr=fp / (fp + tn) if fp + tn else None,
                roc_auc=float(roc_auc_score(y, scores)) if both else None,
                average_precision=float(average_precision_score(y, scores)) if tp + fn else None,
                intervals_95={'recall': wilson_interval(tp, tp + fn),
                              'fpr': wilson_interval(fp, fp + tn)},
                confusion_matrix={'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp})


def evaluate_model(bundle, texts, labels, model_path, repeats=5, benchmark_samples=512):
    scores = score_batches(bundle, texts)
    report = metrics(labels, scores, bundle['threshold'])
    rng = np.random.default_rng(42)
    indices = rng.choice(len(texts), size=min(benchmark_samples, len(texts)), replace=False)
    batch = [texts[i] for i in indices]
    score_texts(bundle, batch[:16])
    durations = []
    for _ in range(repeats):
        start = perf_counter()
        score_texts(bundle, batch)
        durations.append((perf_counter() - start) * 1000 / len(batch))
    report.update(threshold=float(bundle['threshold']),
                  model_bytes=Path(model_path).stat().st_size,
                  model_mib=Path(model_path).stat().st_size / 1024**2,
                  model_sha256=hashlib.sha256(Path(model_path).read_bytes()).hexdigest(),
                  timing={'batch_samples': len(batch), 'repeats': repeats,
                          'median_ms_per_sample': float(np.median(durations)),
                          'includes': 'normalization + TF-IDF + SVM; excludes file/model loading',
                          'kind': 'batch throughput, not single-request latency'})
    return report
