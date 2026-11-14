"""Train TF-IDF + LinearSVC on the local dataset and evaluate the held-out test."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
from time import perf_counter

from core import (HERE, evaluate_model, metrics, score_batches, select_threshold, write_json)
from data import DEFAULT_DATA, load_splits
import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split


def train_model(data_dir=DEFAULT_DATA, output_dir=HERE, seed=42, limit=0,
                max_chars=4000, max_features=20000, max_fpr=.05):
    if not 0 <= max_fpr <= 1 or max_chars <= 0 or max_features < 1 or limit < 0:
        raise ValueError('nieprawidłowe parametry treningu')
    output_dir = Path(output_dir)
    def progress(message):
        print(message, file=sys.stderr, flush=True)
    progress('1/5 Wczytywanie, SHA-256, normalizacja, deduplikacja i podział danych')
    train, validation, test, split_info = load_splits(data_dir, seed, limit, max_chars)
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(analyzer='char', ngram_range=(3, 5), lowercase=False,
                                 min_df=2, max_features=max_features, sublinear_tf=True,
                                 dtype=np.float32)),
        ('svm', LinearSVC(C=1, class_weight='balanced', max_iter=5000, random_state=seed)),
    ])
    progress(f'2/5 Uczenie wyłącznie na train: {len(train)} rekordów')
    start = perf_counter()
    pipeline.fit(train.text, train.label)
    fit_seconds = perf_counter() - start
    progress(f'3/5 Dobór progu na validation: {len(validation)} rekordów')
    bundle = {'format_version': 1, 'pipeline': pipeline, 'max_chars': max_chars,
              'threshold': 0., 'labels': {0: 'benign', 1: 'phishing'}, 'split_info': split_info}
    val_scores = score_batches(bundle, validation.text.tolist())
    bundle['threshold'] = select_threshold(validation.label, val_scores, max_fpr)
    validation_metrics = metrics(validation.label, val_scores, bundle['threshold'])
    if validation_metrics['fpr'] > max_fpr + 1e-12:
        raise RuntimeError('próg nie spełnia budżetu FPR walidacji')
    model_path = output_dir / 'models' / 'pipeline.joblib'
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_path, compress=3)
    # Evaluate the serialized artifact, not just the in-memory estimator.
    bundle = joblib.load(model_path)
    progress(f'4/5 Ocena odłożonego testu: {len(test)} rekordów')
    test_metrics = evaluate_model(bundle, test.text.tolist(), test.label.tolist(), model_path)
    test_metrics['scope'] = 'full held-out test; not used in fitting or threshold selection'
    write_json(output_dir / 'reports' / 'test_metrics.json', test_metrics)
    metadata = dict(format_version=1, architecture='char TF-IDF (3,5) + LinearSVC',
                    threshold=bundle['threshold'], threshold_policy={'objective':'maximum validation recall',
                        'max_validation_fpr':max_fpr, 'tie_break':'lower FPR, then higher threshold'},
                    split_info=split_info, validation=validation_metrics,
                    features=len(pipeline['tfidf'].vocabulary_), fit_seconds=fit_seconds,
                    model_bytes=model_path.stat().st_size,
                    model_sha256=hashlib.sha256(model_path.read_bytes()).hexdigest(),
                    versions={'python':platform.python_version(), 'numpy':np.__version__,
                              'pandas':pd.__version__, 'scipy':scipy.__version__,
                              'scikit-learn':sklearn.__version__, 'joblib':joblib.__version__})
    write_json(model_path.with_name('metadata.json'), metadata)
    count = min(200, len(test))
    sample = test if count == len(test) else train_test_split(
        test, train_size=count, random_state=seed, stratify=test.label)[0]
    examples = output_dir / 'examples' / 'test.jsonl'
    examples.parent.mkdir(parents=True, exist_ok=True)
    with examples.open('w', encoding='utf-8') as handle:
        for row in sample.itertuples():
            handle.write(json.dumps({'id':row.key, 'text':row.text, 'label':int(row.label)},
                                    ensure_ascii=False) + '\n')
    sample_metrics = evaluate_model(bundle, sample.text.tolist(), sample.label.tolist(), model_path)
    sample_metrics['scope'] = '200-record subset of held-out test, not an independent test'
    write_json(output_dir / 'reports' / 'sample_metrics.json', sample_metrics)
    progress('5/5 Zapisano model, metadane, raporty i przykłady testowe')
    return metadata, test_metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=DEFAULT_DATA)
    parser.add_argument('--output-dir', type=Path, default=HERE)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--limit', type=int, default=0, help='0 = pełny zbiór; dodatni limit = eksperyment')
    parser.add_argument('--max-chars', type=int, default=4000)
    parser.add_argument('--max-features', type=int, default=20000)
    parser.add_argument('--max-fpr', type=float, default=.05)
    args = parser.parse_args()
    metadata, report = train_model(**vars(args))
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
