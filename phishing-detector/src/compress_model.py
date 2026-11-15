"""Compress a trusted fitted model; refit only on its original training split."""
import argparse
from pathlib import Path
from time import perf_counter
import warnings

from sklearn.exceptions import ConvergenceWarning

from compression import (DEFAULT_CHAR_KEEP, DEFAULT_WORD_KEEP, DEFAULT_WEIGHT_BITS,
                         compress_pipeline)
from core import load_model
from data import load_splits, matches_split
from train import export_solution


def compress_model(model, data_dir, output_dir, char_keep=DEFAULT_CHAR_KEEP,
                   word_keep=DEFAULT_WORD_KEEP, weight_bits=DEFAULT_WEIGHT_BITS,
                   min_recall=.928):
    source = load_model(model)
    destination = Path(output_dir) / 'models' / 'pipeline.joblib'
    if destination.resolve() == Path(model).resolve():
        raise ValueError('wynik nie może nadpisywać modelu źródłowego')
    expected = source['split_info']
    train, validation, test, info = load_splits(
        data_dir, expected['seed'], expected['limit'], source['max_chars'],
        **expected.get('schema', {}))
    if not matches_split(expected, info):
        raise ValueError('dataset lub podział nie odpowiada modelowi źródłowemu')
    start = perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter('error', ConvergenceWarning)
        pipeline, quantization = compress_pipeline(
            source['pipeline'], train.text, train.label, char_keep, word_keep, weight_bits)
    config = {**source['configuration'], 'compression': dict(
        char_keep=char_keep, word_keep=word_keep, weight_bits=weight_bits,
        feature_selection='top absolute training weights per branch; refit on train')}
    return export_solution(pipeline, validation, test, info, output_dir, config,
                           perf_counter() - start, max_chars=source['max_chars'],
                           min_recall=min_recall, quantization=quantization,
                           fit_seconds_scope='compression refit; reference training excluded')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True, help='zaufany model bazowy przed kompresją')
    parser.add_argument('--input', dest='data_dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--char-keep', type=int, default=DEFAULT_CHAR_KEEP)
    parser.add_argument('--word-keep', type=int, default=DEFAULT_WORD_KEEP)
    parser.add_argument('--weight-bits', type=int, choices=[6, 8, 10, 32], default=DEFAULT_WEIGHT_BITS)
    parser.add_argument('--min-recall', type=float, default=.928)
    args = parser.parse_args()
    try:
        compress_model(**vars(args))
    except (ValueError, OSError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()
