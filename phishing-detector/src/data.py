"""Read the local dataset with integrity verification and reproducible splitting."""
import gzip
import hashlib
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from core import normalize, text_key, read_records

def load_splits(directory=None, seed=42, limit=0, max_chars=4000, **schema):
    if directory is None:
        raise ValueError('wskaż lokalny plik danych przez --input')
    directory = Path(directory)
    if directory.is_dir():
        # Backwards-compatible reconstruction of externally stored legacy archives.
        manifest = json.loads((directory / 'manifest.json').read_text())
        chunks = []
        for part in manifest['parts']:
            chunk = (directory / part['file']).read_bytes()
            if len(chunk) != part['bytes'] or hashlib.sha256(chunk).hexdigest() != part['sha256']:
                raise ValueError(f"uszkodzona część datasetu: {part['file']}")
            chunks.append(chunk)
        raw = gzip.decompress(b''.join(chunks))
        if hashlib.sha256(raw).hexdigest() != manifest['original_sha256']:
            raise ValueError('niezgodna suma SHA-256 oryginalnego JSON')
        records = json.loads(raw)
        if len(records) != manifest['records']:
            raise ValueError('niezgodna liczba rekordów')
        source_hash = manifest['original_sha256']
        source_revision = manifest.get('source_revision')
    else:
        digest = hashlib.sha256()
        with directory.open('rb') as handle:
            for chunk in iter(lambda: handle.read(1024*1024), b''):
                digest.update(chunk)
        source_hash, source_revision = digest.hexdigest(), None
        records = list(read_records(directory, labels=True, allow_empty=True, **schema))
    if not records:
        raise ValueError('pusty zbiór danych')
    rows, invalid = [], 0
    for record in records:
        if type(record.get('label')) is not int or record['label'] not in (0, 1):
            raise ValueError('nieprawidłowa etykieta w zbiorze')
        try:
            text = normalize(record.get('text'), max_chars)
        except ValueError:
            invalid += 1
            continue
        rows.append({'text': text, 'label': record['label'], 'key': text_key(text)})
    if not rows:
        raise ValueError('brak poprawnych wejść po normalizacji')
    frame = pd.DataFrame(rows)
    conflicts = frame.groupby('key').label.nunique()
    conflicts = set(conflicts[conflicts > 1].index)
    conflicting_records = int(frame.key.isin(conflicts).sum())
    frame = frame[~frame.key.isin(conflicts)]
    before = len(frame)
    frame = frame.drop_duplicates('key').reset_index(drop=True)
    retained = len(frame)
    if limit and limit < retained:
        frame, _ = train_test_split(frame, train_size=limit, stratify=frame.label, random_state=seed)
    if set(frame.label) != {0, 1} or frame.label.value_counts().min() < 5:
        raise ValueError('trening wymaga co najmniej 5 różnych wejść każdej klasy po deduplikacji')
    try:
        train, rest = train_test_split(frame, test_size=.4, stratify=frame.label, random_state=seed)
        validation, test = train_test_split(rest, test_size=.5, stratify=rest.label, random_state=seed)
    except ValueError as exc:
        raise ValueError('za mało danych do rozłącznego podziału train/validation/test') from exc
    splits = [part.reset_index(drop=True) for part in (train, validation, test)]
    keys = [set(part.key) for part in splits]
    if keys[0] & keys[1] or keys[0] & keys[2] or keys[1] & keys[2]:
        raise ValueError('przeciek identycznych wejść między podziałami')
    info = dict(source_sha256=source_hash, source_revision=source_revision,
                source_records=len(records), invalid_records=invalid,
                conflicting_records_removed=conflicting_records,
                duplicates_removed=before - retained, retained_records=retained,
                selected_records=len(frame), seed=seed, limit=limit, max_chars=max_chars,
                splits={name: {'samples': len(part), 'phishing': int(part.label.sum()),
                               'keys_sha256': text_key('\n'.join(sorted(part.key)))}
                        for name, part in zip(['train', 'validation', 'test'], splits)})
    if schema:
        info['schema'] = schema
    return *splits, info


def matches_split(expected, actual):
    # Source revision/path need not be known for a raw file; bytes and split must match.
    keys = ('source_sha256', 'source_records', 'seed', 'limit', 'max_chars', 'splits')
    defaults = dict(text_column='text', label_column='label', id_column='id',
                    positive_label='1', negative_label='0', delimiter=',')
    return (all(expected.get(key) == actual.get(key) for key in keys)
            and {**defaults, **expected.get('schema', {})} == {**defaults, **actual.get('schema', {})})
