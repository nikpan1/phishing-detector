"""Train a local TF-IDF pipeline, select a validation threshold, then evaluate test."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
import warnings
from time import perf_counter

from core import (HERE, evaluate_model, metrics, score_batches, select_threshold,
                  select_threshold_for_recall, write_json)
from data import DEFAULT_DATA, load_splits
import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.exceptions import ConvergenceWarning
from modeling import DEFAULT_CONFIG, DEFAULT_RECALL, architecture_name, build_pipeline, feature_count
from sklearn.model_selection import train_test_split


def export_solution(pipeline, validation, test, split_info, output_dir, config,
                    fit_seconds, max_chars=4000, max_fpr=None, min_recall=None):
    output_dir = Path(output_dir)
    if max_fpr is not None and min_recall is not None:
        raise ValueError('wybierz max_fpr albo min_recall')
    if max_fpr is None and min_recall is None:
        min_recall = DEFAULT_RECALL
    bundle = {'format_version': 1, 'pipeline': pipeline, 'max_chars': max_chars,
              'threshold': 0., 'labels': {0: 'benign', 1: 'phishing'},
              'split_info': split_info, 'configuration': config}
    print(f'3/5 Dobór progu na validation: {len(validation)} rekordów', file=sys.stderr, flush=True)
    val_scores = score_batches(bundle, validation.text.tolist())
    if min_recall is not None:
        bundle['threshold'] = select_threshold_for_recall(validation.label, val_scores, min_recall)
        policy = {'objective': 'minimum validation FPR at recall floor',
                  'min_validation_recall': min_recall,
                  'tie_break': 'higher recall, then higher threshold'}
    else:
        bundle['threshold'] = select_threshold(validation.label, val_scores, max_fpr)
        policy = {'objective': 'maximum validation recall', 'max_validation_fpr': max_fpr,
                  'tie_break': 'lower FPR, then higher threshold'}
    validation_metrics = metrics(validation.label, val_scores, bundle['threshold'])
    if min_recall is not None and validation_metrics['recall'] + 1e-12 < min_recall:
        raise RuntimeError('próg nie spełnia minimum czułości walidacji')
    if max_fpr is not None and validation_metrics['fpr'] > max_fpr + 1e-12:
        raise RuntimeError('próg nie spełnia budżetu FPR walidacji')
    model_path = output_dir / 'models' / 'pipeline.joblib'
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_path, compress=3)
    # Evaluate the serialized artifact, not just the in-memory estimator.
    bundle = joblib.load(model_path)
    print(f'4/5 Ocena odłożonego testu: {len(test)} rekordów', file=sys.stderr, flush=True)
    test_metrics = evaluate_model(bundle, test.text.tolist(), test.label.tolist(), model_path)
    test_metrics['scope'] = 'same held-out test as baseline; not used in this optimization selection; not a new blind test'
    write_json(output_dir / 'reports' / 'test_metrics.json', test_metrics)
    metadata = dict(format_version=1, architecture=architecture_name(config), configuration=config,
                    threshold=bundle['threshold'], threshold_policy=policy,
                    split_info=split_info, validation=validation_metrics,
                    features=feature_count(pipeline), fit_seconds=fit_seconds,
                    model_bytes=model_path.stat().st_size,
                    model_sha256=hashlib.sha256(model_path.read_bytes()).hexdigest(),
                    versions={'python':platform.python_version(), 'numpy':np.__version__,
                              'pandas':pd.__version__, 'scipy':scipy.__version__,
                              'scikit-learn':sklearn.__version__, 'joblib':joblib.__version__})
    write_json(model_path.with_name('metadata.json'), metadata)
    count = min(200, len(test))
    sample = test if count == len(test) else train_test_split(
        test, train_size=count, random_state=split_info['seed'], stratify=test.label)[0]
    examples = output_dir / 'examples' / 'test.jsonl'
    examples.parent.mkdir(parents=True, exist_ok=True)
    with examples.open('w', encoding='utf-8') as handle:
        for row in sample.itertuples():
            handle.write(json.dumps({'id':row.key, 'text':row.text, 'label':int(row.label)},
                                    ensure_ascii=False) + '\n')
    sample_metrics = evaluate_model(bundle, sample.text.tolist(), sample.label.tolist(), model_path)
    sample_metrics['scope'] = '200-record subset of held-out test, not an independent test'
    write_json(output_dir / 'reports' / 'sample_metrics.json', sample_metrics)
    print('5/5 Zapisano model, metadane, raporty i przykłady testowe', file=sys.stderr, flush=True)
    return metadata, test_metrics


def train_model(data_dir=DEFAULT_DATA, output_dir=HERE, seed=42, limit=0,
                max_chars=4000, max_features=DEFAULT_CONFIG['max_features'],
                word_features=DEFAULT_CONFIG['word_features'], svm_c=DEFAULT_CONFIG['svm_c'],
                architecture=DEFAULT_CONFIG['architecture'], classifier=DEFAULT_CONFIG['classifier'],
                max_fpr=None, min_recall=None):
    if max_chars <= 0 or limit < 0:
        raise ValueError('nieprawidłowe parametry treningu')
    if max_fpr is not None and min_recall is not None:
        raise ValueError('wybierz max_fpr albo min_recall')
    if max_fpr is not None and not 0 <= max_fpr <= 1:
        raise ValueError('max_fpr musi należeć do [0,1]')
    if min_recall is not None and not 0 < min_recall <= 1:
        raise ValueError('min_recall musi należeć do (0,1]')
    print('1/5 Wczytywanie, SHA-256, normalizacja, deduplikacja i podział danych', file=sys.stderr, flush=True)
    train, validation, test, split_info = load_splits(data_dir, seed, limit, max_chars)
    config = dict(architecture=architecture, max_features=max_features, word_features=word_features,
                  svm_c=svm_c, classifier=classifier)
    pipeline = build_pipeline(**config, seed=seed)
    print(f'2/5 Uczenie wyłącznie na train: {len(train)} rekordów', file=sys.stderr, flush=True)
    start = perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter('error', ConvergenceWarning)
        pipeline.fit(train.text, train.label)
    return export_solution(pipeline, validation, test, split_info, output_dir, config,
                           perf_counter() - start, max_chars, max_fpr, min_recall)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=DEFAULT_DATA)
    parser.add_argument('--output-dir', type=Path, default=HERE)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--limit', type=int, default=0, help='0 = pełny zbiór; dodatni limit = eksperyment')
    parser.add_argument('--max-chars', type=int, default=4000)
    parser.add_argument('--max-features', type=int, default=DEFAULT_CONFIG['max_features'])
    parser.add_argument('--word-features', type=int, default=DEFAULT_CONFIG['word_features'])
    parser.add_argument('--architecture', choices=['char', 'hybrid'], default=DEFAULT_CONFIG['architecture'])
    parser.add_argument('--classifier', choices=['svm', 'logistic'], default=DEFAULT_CONFIG['classifier'])
    parser.add_argument('--svm-c', type=float, default=DEFAULT_CONFIG['svm_c'])
    policy = parser.add_mutually_exclusive_group()
    policy.add_argument('--max-fpr', type=float, help='maksymalizuj recall przy limicie FPR walidacji')
    policy.add_argument('--min-recall', type=float, help='minimalizuj FPR; domyślne minimum recall: 0.928')
    args = parser.parse_args()
    metadata, report = train_model(**vars(args))
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
