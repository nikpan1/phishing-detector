"""Evaluate labels, recall, false positives, model size and CPU batch throughput."""
import argparse
import json
from pathlib import Path

from core import DEFAULT_MODEL, evaluate_model, load_model, read_records, write_json, input_options, record_options
from data import load_splits, matches_split


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, default=DEFAULT_MODEL)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--input', type=Path, help='JSONL/CSV z text i label')
    inputs.add_argument('--dataset', '--data-dir', dest='data_dir', type=Path, help='odtwórz pełny test zgodnie z modelem')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--benchmark-samples', type=int, default=512)
    input_options(parser)
    args = parser.parse_args()
    if args.repeats < 1 or args.benchmark_samples < 1:
        parser.error('parametry benchmarku muszą być dodatnie')
    try:
        if args.output and args.output.resolve() in {
                args.model.resolve(), args.input.resolve() if args.input else None}:
            raise ValueError('raport nie może nadpisywać modelu ani danych wejściowych')
        bundle = load_model(args.model)
        if args.data_dir:
            expected = bundle['split_info']
            _, _, test, actual = load_splits(args.data_dir, expected['seed'],
                                             expected['limit'], bundle['max_chars'], **expected.get('schema', {}))
            if not matches_split(expected, actual):
                raise ValueError('dataset lub podział nie odpowiada metadanym modelu')
            texts, labels = test.text.tolist(), test.label.tolist()
        else:
            records = list(read_records(args.input, labels=True, **record_options(args)))
            texts, labels = [r['text'] for r in records], [r['label'] for r in records]
        report = evaluate_model(bundle, texts, labels, args.model,
                                args.repeats, args.benchmark_samples)
        report['scope'] = 'reconstructed held-out test' if args.data_dir else str(args.input)
        if args.output:
            write_json(args.output, report)
        print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    except (ValueError, OSError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()
