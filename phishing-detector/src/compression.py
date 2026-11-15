"""Feature pruning, mixed-precision quantization and portable model storage."""
from copy import deepcopy

import joblib
import numpy as np
from sklearn.base import clone

from modeling import MODEL_COMPRESSION, compact_pipeline, feature_transformers

DEFAULT_CHAR_KEEP = 10000
DEFAULT_WORD_KEEP = 5000
DEFAULT_WEIGHT_BITS = 6


def prune_pipeline(pipeline, char_keep=0, word_keep=0):
    """Select top absolute SVM weights per branch, preserving feature order.

    Zero means keep all. The caller must refit the classifier on train because
    dropping terms changes the per-branch TF-IDF normalization.
    """
    if char_keep < 0 or word_keep < 0:
        raise ValueError('liczby zachowanych cech nie mogą być ujemne')
    result = compact_pipeline(pipeline)
    weights = result.steps[-1][1].coef_
    if weights.shape[0] != 1:
        raise ValueError('kompresja obsługuje klasyfikator binarny')
    offset, selected = 0, []
    for name, transformer in feature_transformers(result):
        count = len(transformer.vocabulary_)
        keep = word_keep if name == 'word' else char_keep
        keep = min(keep or count, count)
        local_weights = weights[0, offset:offset + count]
        indices = np.sort(np.argsort(-np.abs(local_weights), kind='stable')[:keep])
        terms = {index: term for term, index in transformer.vocabulary_.items()}
        transformer.vocabulary_ = {terms[int(index)]: i for i, index in enumerate(indices)}
        transformer.idf_ = transformer.idf_[indices]
        transformer._tfidf.n_features_in_ = len(indices)
        selected.extend((offset + indices).tolist())
        offset += count
    classifier = result.steps[-1][1]
    classifier.coef_ = weights[:, selected].copy()
    classifier.n_features_in_ = len(selected)
    return result


def quantize_pipeline(pipeline, bits=8):
    """Round the central 99.9% of weights; retain rare outliers exactly.

    Values remain float32, so inference uses standard sklearn with no decoding
    or custom estimator. The small shared value set compresses well with LZMA.
    """
    if bits not in (6, 8, 10, 32):
        raise ValueError('weight_bits musi wynosić 6, 8, 10 lub 32')
    result = compact_pipeline(pipeline)
    weights = result.steps[-1][1].coef_
    scale = None
    outliers = 0
    if bits != 32:
        scale = float(np.quantile(np.abs(weights), .999) / (2**(bits - 1) - 1))
        if scale > 0:
            limit = scale * (2**(bits - 1) - 1)
            mask = np.abs(weights) <= limit
            result.steps[-1][1].coef_ = np.where(
                mask, np.rint(weights / scale) * scale, weights).astype(np.float32)
            outliers = int((~mask).sum())
    return result, {'bits': bits, 'scale': scale, 'outlier_quantile': .999,
                    'outliers_preserved': outliers, 'runtime_dtype': 'float32'}


def compress_pipeline(pipeline, texts, labels, char_keep=0, word_keep=0, weight_bits=8):
    """Prune/refit on train only, then quantize before validation calibration."""
    if char_keep or word_keep:
        pipeline = prune_pipeline(pipeline, char_keep, word_keep)
        classifier = clone(pipeline.steps[-1][1])
        classifier.fit(pipeline[:-1].transform(texts), labels)
        pipeline.steps[-1] = (pipeline.steps[-1][0], classifier)
    return quantize_pipeline(pipeline, weight_bits)


def dump_model(bundle, path):
    """Store vocabularies as ordered tuples instead of redundant term/index pairs.

    The artifact contains only standard sklearn/numpy objects and also works
    with plain joblib.load(...)["pipeline"].decision_function(...).
    """
    packed = deepcopy(bundle)
    for _, transformer in feature_transformers(packed['pipeline']):
        if hasattr(transformer, 'vocabulary_'):
            transformer.vocabulary = tuple(term for term, _ in sorted(
                transformer.vocabulary_.items(), key=lambda item: item[1]))
            del transformer.vocabulary_
    return joblib.dump(packed, path, compress=MODEL_COMPRESSION)
