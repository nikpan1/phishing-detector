"""Boundary and end-to-end checks for the delivered local model."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC))
from core import (DEFAULT_MODEL, load_model, metrics, normalize, read_records,
                  score_texts, select_threshold)


class PreprocessingTests(unittest.TestCase):
    def test_unicode_cap_and_idempotence(self):
        text = normalize('İ' * 5000)
        self.assertLessEqual(len(text), 4000)
        self.assertEqual(text, normalize(text))
        self.assertEqual(normalize('  Account\n\tUPDATE '), 'account update')

    def test_reject_invalid_or_empty_effective_input(self):
        for value in [None, 123, '', '   ', ' ' * 4000 + 'hello']:
            with self.subTest(value_type=type(value).__name__), self.assertRaises(ValueError):
                normalize(value)

    def test_tied_scores_respect_fpr_budget(self):
        labels = np.array([0, 1, 0, 1])
        scores = np.array([0., 0., 1., 1.])
        threshold = select_threshold(labels, scores, .5)
        report = metrics(labels, scores, threshold)
        self.assertEqual(report['fpr'], .5)
        self.assertEqual(report['recall'], .5)
        threshold = select_threshold(labels, scores, 0)
        self.assertEqual(metrics(labels, scores, threshold)['fpr'], 0)

    def test_single_class_has_no_fake_auc(self):
        report = metrics([0, 0], [-1., 1.], 0.)
        self.assertIsNone(report['roc_auc'])
        self.assertIsNone(report['recall'])
        self.assertEqual(report['fpr'], .5)

    def test_json_and_csv_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'test.jsonl'
            path.write_text(json.dumps({'text':'hello', 'label':True}) + '\n')
            with self.assertRaises(ValueError):
                list(read_records(path, labels=True))
            csv_path = Path(directory) / 'test.csv'
            csv_path.write_text('text,label\n"Hello, world",0\n', encoding='utf-8')
            rows = list(read_records(csv_path, labels=True))
            self.assertEqual(rows[0]['label'], 0)
            self.assertEqual(rows[0]['text'], 'Hello, world')


class DeliveredArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = load_model()
        cls.metadata = json.loads((SRC / 'models' / 'metadata.json').read_text())
        cls.examples = list(read_records(SRC / 'examples' / 'test.jsonl', labels=True))

    def test_artifact_digest_and_validation_constraint(self):
        self.assertEqual(hashlib.sha256(DEFAULT_MODEL.read_bytes()).hexdigest(),
                         self.metadata['model_sha256'])
        self.assertLessEqual(self.metadata['validation']['fpr'],
                             self.metadata['threshold_policy']['max_validation_fpr'] + 1e-12)

    def test_batch_and_single_predictions_agree(self):
        texts = [r['text'] for r in self.examples[:8]]
        batched = score_texts(self.bundle, texts)
        single = [score_texts(self.bundle, [text])[0] for text in texts]
        np.testing.assert_allclose(batched, single, atol=1e-6)

    def test_sample_report_matches_shipped_model(self):
        scores = score_texts(self.bundle, [r['text'] for r in self.examples])
        actual = metrics([r['label'] for r in self.examples], scores, self.bundle['threshold'])
        recorded = json.loads((SRC / 'reports' / 'sample_metrics.json').read_text())
        self.assertEqual(actual['confusion_matrix'], recorded['confusion_matrix'])
        self.assertEqual(actual['recall'], recorded['recall'])

    def test_jsonl_cli_handles_multiple_batches(self):
        output = subprocess.check_output([sys.executable, str(SRC / 'predict.py'),
            '--input', str(SRC / 'examples' / 'test.jsonl'), '--batch-size', '17'], text=True)
        predictions = [json.loads(line) for line in output.splitlines()]
        self.assertEqual(len(predictions), len(self.examples))
        scores = score_texts(self.bundle, [r['text'] for r in self.examples])
        np.testing.assert_allclose([r['score'] for r in predictions], scores, atol=1e-6)
        self.assertEqual([r['id'] for r in predictions], [r['id'] for r in self.examples])

    def test_cli_rejects_overwriting_input(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'input.csv'
            content = 'text\nhello\n'
            path.write_text(content)
            result = subprocess.run([sys.executable, str(SRC / 'predict.py'),
                '--input', str(path), '--output', str(path)], capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(path.read_text(), content)


if __name__ == '__main__':
    unittest.main()
