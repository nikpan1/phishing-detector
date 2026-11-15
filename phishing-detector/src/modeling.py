"""Portable sklearn pipelines; no custom objects are needed in model artifacts."""
from copy import deepcopy

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

DEFAULT_CONFIG = dict(architecture='hybrid', max_features=60000, word_features=30000,
                      svm_c=.25, classifier='svm')
DEFAULT_RECALL = .928
MODEL_COMPRESSION = ('lzma', 6)


def feature_transformers(pipeline):
    return (pipeline['features'].transformer_list if 'features' in pipeline.named_steps
            else [('tfidf', pipeline['tfidf'])])


def restore_vocabularies(pipeline):
    """Materialize the standard sklearn ordered-vocabulary representation."""
    for _, transformer in feature_transformers(pipeline):
        if not hasattr(transformer, 'vocabulary_'):
            transformer.vocabulary_ = {term: index for index, term
                                       in enumerate(transformer.vocabulary)}
            transformer.fixed_vocabulary_ = True
        # Avoid keeping a second copy after loading a packed artifact.
        transformer.vocabulary = None
    return pipeline


def compact_pipeline(pipeline):
    """Keep all features; use native vocabulary indices and float32 weights.

    Return a copy so callers can compare against the original. Compact before
    validation/threshold selection: float32 introduces small rounding errors.
    The artifact still contains only standard sklearn/numpy objects.
    """
    pipeline = restore_vocabularies(deepcopy(pipeline))
    for _, transformer in feature_transformers(pipeline):
        transformer.vocabulary_ = {term: int(index) for term, index
                                   in sorted(transformer.vocabulary_.items())}
    classifier = pipeline.steps[-1][1]
    classifier.coef_ = classifier.coef_.astype(np.float32)
    classifier.intercept_ = classifier.intercept_.astype(np.float32)
    return pipeline


def vectorizer(analyzer, max_features):
    return TfidfVectorizer(analyzer=analyzer,
                          ngram_range=(3, 5) if analyzer == 'char' else (1, 2),
                          lowercase=False, min_df=2, max_features=max_features,
                          sublinear_tf=True, dtype=np.float32)


def estimator(classifier='svm', svm_c=1., seed=42):
    if classifier == 'svm':
        return LinearSVC(C=svm_c, class_weight='balanced', max_iter=10000, random_state=seed)
    if classifier == 'logistic':
        return LogisticRegression(C=svm_c, class_weight='balanced', solver='liblinear',
                                  max_iter=2000, random_state=seed)
    raise ValueError('nieznany klasyfikator')


def build_pipeline(architecture=DEFAULT_CONFIG['architecture'], max_features=DEFAULT_CONFIG['max_features'],
                   word_features=DEFAULT_CONFIG['word_features'], svm_c=DEFAULT_CONFIG['svm_c'],
                   classifier=DEFAULT_CONFIG['classifier'], seed=42):
    if max_features < 1 or word_features < 1 or not np.isfinite(svm_c) or svm_c <= 0:
        raise ValueError('liczby cech i C muszą być dodatnie')
    char = vectorizer('char', max_features)
    if architecture == 'char':
        feature_step = ('tfidf', char)
    elif architecture == 'hybrid':
        feature_step = ('features', FeatureUnion([
            ('char', char), ('word', vectorizer('word', word_features))], n_jobs=1))
    else:
        raise ValueError('nieznana architektura')
    return Pipeline([feature_step, ('svm' if classifier == 'svm' else 'classifier',
                                   estimator(classifier, svm_c, seed))])


def feature_count(pipeline):
    return sum(len(transformer.vocabulary_) if hasattr(transformer, 'vocabulary_')
               else len(transformer.vocabulary)
               for _, transformer in feature_transformers(pipeline))


def architecture_name(config):
    features = 'char (3,5) + word (1,2)' if config['architecture'] == 'hybrid' else 'char (3,5)'
    classifier = 'LinearSVC' if config['classifier'] == 'svm' else 'LogisticRegression'
    return f'{features} TF-IDF + {classifier}'
