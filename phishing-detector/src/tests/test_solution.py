"""Boundary and end-to-end checks for the delivered local model."""
import hashlib
import io
import csv
import gzip
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
from modeling import MODEL_COMPRESSION, build_pipeline, compact_pipeline, feature_count
from data import load_splits, matches_split
from compression import dump_model, prune_pipeline, quantize_pipeline


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

    def test_compact_artifact_stays_below_quarter_previous_size(self):
        self.assertLess(DEFAULT_MODEL.stat().st_size, 714158 / 4)
        self.assertEqual(feature_count(self.bundle['pipeline']), 15000)

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
                original_scores = score_texts(bundle, texts)
                original_dtype = pipeline['svm'].coef_.dtype
                compact = {**bundle, 'pipeline': compact_pipeline(pipeline)}
                self.assertEqual(pipeline['svm'].coef_.dtype, original_dtype)
                self.assertEqual(compact['pipeline']['svm'].coef_.dtype, np.float32)
                joblib.dump(compact, path, compress=MODEL_COMPRESSION)
                restored = load_model(path)
                np.testing.assert_allclose(original_scores, score_texts(restored, texts),
                                           rtol=1e-5, atol=3e-6)
                np.testing.assert_array_equal(original_scores >= bundle['threshold'],
                                              score_texts(restored, texts) >= restored['threshold'])


class CompressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.texts = ['meeting agenda tomorrow', 'team meeting schedule',
                     'meeting notes and agenda', 'team schedule tomorrow',
                     'urgent verify your account', 'confirm account password',
                     'verify password immediately', 'urgent account confirmation']
        cls.pipeline = build_pipeline(max_features=500, word_features=100)
        cls.pipeline.fit(cls.texts, [0, 0, 0, 0, 1, 1, 1, 1])

    def test_pruned_features_match_renormalized_original_columns(self):
        from scipy.sparse import hstack
        from sklearn.preprocessing import normalize as normalize_matrix
        for text in [self.texts, ['unknownwords', '']]:
            pruned = prune_pipeline(self.pipeline, char_keep=15, word_keep=5)
            expected = []
            for (_, original), (_, reduced) in zip(
                    self.pipeline['features'].transformer_list,
                    pruned['features'].transformer_list):
                terms = sorted(reduced.vocabulary_, key=reduced.vocabulary_.get)
                columns = [original.vocabulary_[term] for term in terms]
                expected.append(normalize_matrix(original.transform(text)[:, columns]))
            actual = pruned['features'].transform(text)
            np.testing.assert_allclose(actual.toarray(), hstack(expected).toarray(), atol=1e-6)
            self.assertEqual(pruned['svm'].n_features_in_, 20)
            self.assertGreater(feature_count(self.pipeline), 20)

    def test_packed_artifact_works_with_plain_joblib_and_standard_loader(self):
        pipeline, _ = quantize_pipeline(self.pipeline, 8)
        bundle = dict(format_version=1, pipeline=pipeline, max_chars=4000, threshold=.1)
        original_scores = score_texts(bundle, self.texts)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'packed.joblib'
            dump_model(bundle, path)
            for loaded in [joblib.load(path), load_model(path)]:
                self.assertEqual(feature_count(loaded['pipeline']), feature_count(pipeline))
                np.testing.assert_array_equal(score_texts(loaded, self.texts), original_scores)
            self.assertTrue(hasattr(pipeline['features'].transformer_list[0][1], 'vocabulary_'))

    def test_quantization_preserves_outliers_and_bounds_rounding(self):
        original = self.pipeline['svm'].coef_.astype(np.float32)
        for bits in [6, 8, 10, 32]:
            pipeline, info = quantize_pipeline(self.pipeline, bits)
            actual = pipeline['svm'].coef_
            if bits == 32:
                np.testing.assert_array_equal(actual, original)
            else:
                limit = info['scale'] * (2**(bits - 1) - 1)
                outliers = np.abs(original) > limit
                np.testing.assert_array_equal(actual[outliers], original[outliers])
                self.assertLessEqual(float(np.max(np.abs(actual - original))),
                                     info['scale'] / 2 + 1e-6)
        with self.assertRaises(ValueError):
            quantize_pipeline(self.pipeline, 3)
        with self.assertRaises(ValueError):
            prune_pipeline(self.pipeline, -1, 0)

    def test_compression_search_does_not_access_test_split(self):
        import search_compression
        train = pd.DataFrame({'text': self.texts, 'label': [0, 0, 0, 0, 1, 1, 1, 1]})
        info = {'seed': 42, 'limit': 0}
        bundle = dict(pipeline=self.pipeline, threshold=0., max_chars=4000, split_info=info)
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()), \
                patch.object(search_compression, 'load_model', return_value=bundle), \
                patch.object(search_compression, 'matches_split', return_value=True), \
                patch.object(search_compression, 'load_splits',
                             return_value=(train, train.copy(), object(), info)):
            result = search_compression.search('unused', 'unused', directory,
                                               budgets=[(15, 5)], bits_options=[8, 32])
            report = json.loads((Path(directory) / 'candidates.json').read_text())
            self.assertFalse(report['test_used'])
            self.assertEqual(len(report['rows']), 2)
            eligible = [row for row in report['rows'] if row['eligible']]
            self.assertEqual(result['model_bytes'], min(row['model_bytes'] for row in eligible))


class ExternalDataTests(unittest.TestCase):
    @staticmethod
    def records():
        return [{'message': f'team meeting agenda reference {i}', 'kind': 'safe'} for i in range(30)] + [
            {'message': f'urgent verify account password reference {i}', 'kind': 'phish'} for i in range(30)]

    def test_external_formats_and_custom_schema(self):
        options = dict(text_column='message', label_column='kind',
                       positive_label='phish', negative_label='safe', delimiter=';')
        rows = self.records()
        with tempfile.TemporaryDirectory() as directory:
            for suffix in ['.csv', '.json', '.jsonl', '.json.gz', '.csv.gz', '.jsonl.gz']:
                with self.subTest(suffix=suffix):
                    path = Path(directory) / ('external' + suffix)
                    opener = gzip.open if suffix.endswith('.gz') else open
                    with opener(path, 'wt', encoding='utf-8', newline='') as handle:
                        if '.csv' in suffix:
                            writer = csv.DictWriter(handle, fieldnames=['message','kind'], delimiter=';')
                            writer.writeheader(); writer.writerows(rows)
                        elif '.jsonl' in suffix:
                            handle.writelines(json.dumps(row) + '\n' for row in rows)
                        else:
                            json.dump(rows, handle)
                    train, val, test, info = load_splits(path, **options)
                    self.assertEqual([len(train),len(val),len(test)], [36,12,12])
                    self.assertEqual(info['source_records'], 60)
                    self.assertEqual(set(train.label), {0,1})
                    self.assertTrue(set(train.key).isdisjoint(test.key))
                    altered = {**info, 'schema': {**info['schema'], 'positive_label':'safe'}}
                    self.assertFalse(matches_split(info, altered))

    def test_train_predict_and_evaluate_external_file_outside_checkout(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / 'external.csv'
            with dataset.open('w', encoding='utf-8', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=['message','kind'], delimiter=';')
                writer.writeheader(); writer.writerows(self.records())
            mapping = ['--text-column','message','--label-column','kind',
                       '--positive-label','phish','--negative-label','safe','--delimiter',';']
            result = subprocess.run([sys.executable, str(SRC / 'train.py'), '--input', str(dataset),
                '--output-dir', str(root / 'model'), '--max-features','500','--word-features','200',
                *mapping], cwd=root, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            model = root / 'model' / 'models' / 'pipeline.joblib'
            self.assertTrue(model.exists())
            output = subprocess.check_output([sys.executable, str(SRC / 'predict.py'), '--model',str(model),
                '--input',str(dataset), *mapping], cwd=root, text=True)
            self.assertEqual(len(output.splitlines()),60)
            report = json.loads(subprocess.check_output([sys.executable,str(SRC / 'evaluate.py'),
                '--model',str(model),'--input',str(dataset),*mapping], cwd=root, text=True))
            self.assertEqual(report['samples'],60)
            reconstructed = json.loads(subprocess.check_output([sys.executable,str(SRC / 'evaluate.py'),
                '--model',str(model),'--dataset',str(dataset)], cwd=root, text=True))
            self.assertEqual(reconstructed['samples'],12)

    def test_rejects_insufficient_training_data(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'tiny.json'
            path.write_text(json.dumps([{'text':'hello','label':0},{'text':'account','label':1}]))
            with self.assertRaisesRegex(ValueError, 'co najmniej 5'):
                load_splits(path)


if __name__ == '__main__':
    unittest.main()
