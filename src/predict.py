"""Predict phishing locally from --text or a JSONL/CSV file."""
import argparse
from contextlib import nullcontext
from itertools import islice
import json
from pathlib import Path
import sys

from core import DEFAULT_MODEL, load_model, read_records, score_texts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, default=DEFAULT_MODEL)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--text')
    inputs.add_argument('--input', type=Path)
    parser.add_argument('--output', type=Path, help='JSONL; domyślnie stdout')
    parser.add_argument('--batch-size', type=int, default=256)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error('--batch-size musi być dodatni')
    try:
        if args.output and args.output.resolve() in {
                args.model.resolve(), args.input.resolve() if args.input else None}:
            raise ValueError('plik wyjściowy nie może nadpisywać wejścia ani modelu')
        bundle = load_model(args.model)
        records = iter([{'id':1, 'text':args.text}]) if args.text is not None else read_records(args.input)
        output = args.output.open('w', encoding='utf-8') if args.output else nullcontext(sys.stdout)
        count = 0
        with output as handle:
            while batch := list(islice(records, args.batch_size)):
                scores = score_texts(bundle, [r['text'] for r in batch])
                for record, score in zip(batch, scores):
                    label = int(score >= bundle['threshold'])
                    handle.write(json.dumps({'id':record['id'], 'label':label,
                        'prediction':bundle['labels'][label], 'score':float(score),
                        'threshold':bundle['threshold']}, ensure_ascii=False, allow_nan=False) + '\n')
                    count += 1
        if not count:
            raise ValueError('pusty plik wejściowy')
    except (ValueError, OSError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()
