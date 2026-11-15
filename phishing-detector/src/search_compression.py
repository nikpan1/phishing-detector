"""Choose compression on train/validation only; never score test candidates."""
import argparse
from pathlib import Path
import warnings

import numpy as np
from scipy.sparse import hstack
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import normalize

from compression import dump_model, prune_pipeline, quantize_pipeline
from core import load_model, metrics, select_threshold_for_recall, write_json
from data import load_splits, matches_split
from modeling import feature_transformers

FEATURE_BUDGETS = ((60000, 30000), (40000, 20000), (30000, 15000),
                   (20000, 10000), (10000, 5000))
WEIGHT_BITS = (32, 10, 8, 6)


def search(model, data_dir, output_dir, recall_loss=.005, fpr_increase=.003,
           budgets=FEATURE_BUDGETS, bits_options=WEIGHT_BITS):
    if not 0 <= recall_loss < 1 or not 0 <= fpr_increase < 1:
        raise ValueError('limity utraty jakości muszą należeć do [0,1)')
    source = load_model(model)
    expected = source['split_info']
    train, validation, test, info = load_splits(
        data_dir, expected['seed'], expected['limit'], source['max_chars'],
        **expected.get('schema', {}))
    del test
    if not matches_split(expected, info):
        raise ValueError('dataset lub podział nie odpowiada modelowi źródłowemu')
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    original = source['pipeline']
    X = original[:-1].transform(train.text)
    V = original[:-1].transform(validation.text)
    baseline = metrics(validation.label, original.steps[-1][1].decision_function(V), source['threshold'])
    recall_floor = max(np.finfo(float).eps, baseline['recall'] - recall_loss)
    fpr_ceiling = min(1., baseline['fpr'] + fpr_increase)
    rows = []
    for char_keep, word_keep in budgets:
        reduced = prune_pipeline(original, char_keep, word_keep)
        train_blocks, validation_blocks, offset = [], [], 0
        for (_, before), (_, after) in zip(feature_transformers(original), feature_transformers(reduced)):
            terms = sorted(after.vocabulary_, key=after.vocabulary_.get)
            indices = np.array([before.vocabulary_[term] for term in terms]) + offset
            train_blocks.append(normalize(X[:, indices], copy=True))
            validation_blocks.append(normalize(V[:, indices], copy=True))
            offset += len(before.vocabulary_)
        A, B = hstack(train_blocks, format='csr'), hstack(validation_blocks, format='csr')
        if A.shape[1] != X.shape[1]:
            with warnings.catch_warnings():
                warnings.simplefilter('error', ConvergenceWarning)
                classifier = clone(original.steps[-1][1]).fit(A, train.label)
            reduced.steps[-1] = (reduced.steps[-1][0], classifier)
        for bits in bits_options:
            pipeline, quantization = quantize_pipeline(reduced, bits)
            scores = pipeline.steps[-1][1].decision_function(B)
            threshold = select_threshold_for_recall(validation.label, scores, recall_floor)
            result = metrics(validation.label, scores, threshold)
            name = f'char{char_keep}-word{word_keep}-q{bits}'
            path = output_dir / f'{name}.joblib'
            if path.resolve() == Path(model).resolve():
                raise ValueError('wariant nie może nadpisywać modelu źródłowego')
            configuration = {**source.get('configuration', {}), 'compression': dict(
                char_keep=char_keep, word_keep=word_keep, weight_bits=bits,
                feature_selection='top absolute training weights per branch; refit on train')}
            dump_model({**source, 'pipeline': pipeline, 'threshold': threshold,
                        'configuration': configuration}, path)
            row = dict(name=name, char_keep=char_keep, word_keep=word_keep, bits=bits,
                       scale=quantization['scale'], threshold=threshold,
                       model_bytes=path.stat().st_size, **result)
            row['eligible'] = result['recall'] >= recall_floor and result['fpr'] <= fpr_ceiling
            rows.append(row)
            print(f"{name}: {row['model_bytes']} B, recall={row['recall']:.5f}, "
                  f"FPR={row['fpr']:.5f}, eligible={row['eligible']}", flush=True)
    report = dict(recall_floor=recall_floor, fpr_ceiling=fpr_ceiling,
                  baseline=baseline, rows=rows, test_used=False)
    write_json(output_dir / 'candidates.json', report)
    eligible = [row for row in rows if row['eligible']]
    if not eligible:
        raise ValueError('żaden wariant nie spełnia limitów jakości walidacyjnej')
    selected = min(eligible, key=lambda row: (row['model_bytes'], row['fpr'], -row['recall']))
    write_json(output_dir / 'selected.json', selected)
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--input', dest='data_dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--recall-loss', type=float, default=.005)
    parser.add_argument('--fpr-increase', type=float, default=.003)
    args = parser.parse_args()
    try:
        search(**vars(args))
    except (ValueError, OSError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()
