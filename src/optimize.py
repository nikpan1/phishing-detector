"""Compare eight fixed candidates using train/validation only; do not evaluate test."""
import argparse
import json
from pathlib import Path
import sys
from time import perf_counter
import warnings

from core import HERE, metrics, score_batches, select_threshold_for_recall, write_json
from data import DEFAULT_DATA, load_splits
from modeling import DEFAULT_RECALL, estimator, vectorizer
import joblib
from scipy.sparse import hstack
from sklearn.exceptions import ConvergenceWarning
from sklearn.pipeline import FeatureUnion, Pipeline


def search(data_dir=DEFAULT_DATA, output_dir=HERE / 'experiments' / 'optimization',
           min_recall=DEFAULT_RECALL, baseline_model=None):
    if not 0 < min_recall <= 1:
        raise ValueError('min_recall musi należeć do (0,1]')
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train, validation, test, split_info = load_splits(data_dir)
    del test  # no candidate scores, predictions or metrics are computed on test
    rows = []

    def record(name, pipeline, fit_s, config, reused=False):
        bundle = dict(format_version=1, pipeline=pipeline, threshold=0., max_chars=4000,
                      labels={0:'benign', 1:'phishing'}, split_info=split_info, configuration=config)
        scores = score_batches(bundle, validation.text.tolist())
        bundle['threshold'] = select_threshold_for_recall(validation.label, scores, min_recall)
        row = dict(name=name, parameters=config, fit_s=fit_s,
                   reused_pretrained_baseline=reused, threshold=bundle['threshold'],
                   **metrics(validation.label, scores, bundle['threshold']))
        joblib.dump(bundle, output_dir / f'{name}.joblib', compress=3)
        row['model_bytes'] = (output_dir / f'{name}.joblib').stat().st_size
        rows.append(row)
        write_json(output_dir / 'validation_candidates.json', {
            'target_recall':min_recall, 'test_scored_during_search':False,
            'split_info':split_info, 'rows':rows})
        print(json.dumps(row), flush=True)

    def fit_classifier(X, classifier, c):
        model = estimator(classifier, c)
        start = perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter('error', ConvergenceWarning)
            model.fit(X, train.label)
        return model, perf_counter() - start

    config = dict(architecture='char', max_features=20000, word_features=30000,
                  svm_c=1., classifier='svm')
    if baseline_model:
        bundle = joblib.load(baseline_model)
        pipeline = bundle['pipeline']
        if (bundle['split_info'] != split_info or 'tfidf' not in pipeline.named_steps
                or pipeline['tfidf'].analyzer != 'char'
                or pipeline['tfidf'].max_features != 20000
                or pipeline['tfidf'].ngram_range != (3, 5)
                or pipeline['svm'].C != 1. or pipeline['svm'].class_weight != 'balanced'):
            raise ValueError('cache baseline nie odpowiada oryginalnemu modelowi i podziałowi')
        record('baseline-char20k', pipeline, None, config, reused=True)
    else:
        print('Fit baseline char20k on train', flush=True)
        start = perf_counter()
        char = vectorizer('char', 20000)
        X = char.fit_transform(train.text)
        feature_s = perf_counter() - start
        model, fit_s = fit_classifier(X, 'svm', 1.)
        record('baseline-char20k', Pipeline([('tfidf',char), ('svm',model)]),
               feature_s + fit_s, config)
        del X

    print('Fit char60k on train', flush=True)
    start = perf_counter()
    char = vectorizer('char', 60000)
    Xc = char.fit_transform(train.text)
    char_s = perf_counter() - start
    for c in [.25, 1., 4.]:
        model, fit_s = fit_classifier(Xc, 'svm', c)
        config = dict(architecture='char', max_features=60000, word_features=30000,
                      svm_c=c, classifier='svm')
        record(f'char60k-c{c:g}', Pipeline([('tfidf',char), ('svm',model)]), char_s + fit_s, config)

    print('Fit word30k on train', flush=True)
    start = perf_counter()
    word = vectorizer('word', 30000)
    Xw = word.fit_transform(train.text)
    word_s = perf_counter() - start
    union = FeatureUnion([('char',char), ('word',word)], n_jobs=1)
    X = hstack([Xc, Xw], format='csr')
    for classifier, c in [('svm',.25), ('svm',1.), ('svm',4.), ('logistic',4.)]:
        model, fit_s = fit_classifier(X, classifier, c)
        config = dict(architecture='hybrid', max_features=60000, word_features=30000,
                      svm_c=c, classifier=classifier)
        name = f'hybrid90k-c{c:g}' if classifier == 'svm' else 'hybrid90k-logistic-c4'
        record(name, Pipeline([('features',union),
                              ('svm' if classifier == 'svm' else 'classifier',model)]),
               char_s + word_s + fit_s, config)
    selected = min(rows, key=lambda row:(row['fpr'], -row['recall'], row['model_bytes']))
    write_json(output_dir / 'selected.json', selected)
    print('SELECTED ' + selected['name'], flush=True)
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=DEFAULT_DATA)
    parser.add_argument('--output-dir', type=Path, default=HERE / 'experiments' / 'optimization')
    parser.add_argument('--min-recall', type=float, default=DEFAULT_RECALL)
    parser.add_argument('--baseline-model', type=Path, help='opcjonalnie oryginalny model char20k')
    args = parser.parse_args()
    search(**vars(args))


if __name__ == '__main__':
    main()
