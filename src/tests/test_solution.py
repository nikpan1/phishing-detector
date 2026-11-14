"""Boundary and end-to-end checks for the delivered local model."""
import hashlib
import io
from contextlib import redirect_stdout
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import joblib
import pandas as pd

SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC))
from core import (DEFAULT_MODEL, load_model, metrics, normalize, read_records,
                  score_texts, select_threshold, select_threshold_for_recall, wilson_interval)
from modeling import build_pipeline, feature_count


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

    def test_recall_floor_does_not_split_tied_scores(self):
        labels = np.array([0, 1, 0, 1])
        scores = np.array([0., 0., 1., 1.])
        threshold = select_threshold_for_recall(labels, scores, .5)
        report = metrics(labels, scores, threshold)
        self.assertEqual(report['recall'], .5)
        self.assertEqual(report['fpr'], .5)
        threshold = select_threshold_for_recall(labels, scores, 1.)
        self.assertEqual(metrics(labels, scores, threshold)['recall'], 1.)
        with self.assertRaises(ValueError):
            select_threshold_for_recall(labels, scores, 0.)

    def test_uncertainty_for_zero_events_is_not_zero(self):
        lower, upper = wilson_interval(0, 100)
        self.assertAlmostEqual(lower, 0.)
        self.assertAlmostEqual(upper, .036993498, places=7)
        self.assertIsNone(wilson_interval(0, 0))

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
        policy = self.metadata['threshold_policy']
        if 'min_validation_recall' in policy:
            self.assertGreaterEqual(self.metadata['validation']['recall'] + 1e-12,
                                    policy['min_validation_recall'])
        else:
            self.assertLessEqual(self.metadata['validation']['fpr'],
                                 policy['max_validation_fpr'] + 1e-12)

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


class TrainingRecipeTests(unittest.TestCase):
    def test_search_scores_only_validation_and_records_all_candidates(self):
        import optimize
        train = pd.DataFrame({'text': ['team meeting tomorrow', 'meeting notes agenda',
                                      'tomorrow team notes', 'team agenda meeting',
                                      'urgent verify account', 'confirm account password',
                                      'verify password urgent', 'urgent account confirmation'],
                              'label': [0, 0, 0, 0, 1, 1, 1, 1]})
        validation = pd.DataFrame({'text': ['meeting agenda today', 'team meeting notes',
                                           'verify account password', 'urgent verify password'],
                                  'label': [0, 0, 1, 1]})
        def validation_only(bundle, texts):
            self.assertEqual(texts, validation.text.tolist())
            return score_texts(bundle, texts)
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()), \
                patch.object(optimize, 'load_splits', return_value=(train, validation, object(), {'seed':42})), \
                patch.object(optimize, 'score_batches', side_effect=validation_only):
            selected = optimize.search(output_dir=directory)
            report = json.loads((Path(directory) / 'validation_candidates.json').read_text())
            self.assertEqual(len(report['rows']), 8)
            self.assertFalse(report['test_scored_during_search'])
            self.assertEqual(selected['fpr'], min(row['fpr'] for row in report['rows']))

    def test_both_feature_recipes_survive_serialization(self):
        texts = ['meeting agenda tomorrow', 'team meeting schedule',
                 'meeting notes and agenda', 'team schedule tomorrow',
                 'urgent verify your account', 'confirm account password',
                 'verify password immediately', 'urgent account confirmation']
        labels = [0, 0, 0, 0, 1, 1, 1, 1]
        for architecture in ['char', 'hybrid']:
            with self.subTest(architecture=architecture), tempfile.TemporaryDirectory() as directory:
                pipeline = build_pipeline(architecture=architecture, max_features=500,
                                          word_features=100, svm_c=.25, classifier='svm')
                pipeline.fit(texts, labels)
                self.assertGreater(feature_count(pipeline), 0)
                bundle = dict(format_version=1, pipeline=pipeline, threshold=.1, max_chars=4000)
                path = Path(directory) / 'model.joblib'
                joblib.dump(bundle, path)
                restored = load_model(path)
                np.testing.assert_allclose(score_texts(bundle, texts), score_texts(restored, texts))


if __name__ == '__main__':
    unittest.main()
