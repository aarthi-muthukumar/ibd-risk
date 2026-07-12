from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def evaluate_models(y_true, y_probability, sample_weight=None, **kwargs: Any) -> dict:
    y = np.asarray(y_true)
    p = np.asarray(y_probability)
    return {
        'n': int(len(y)),
        'events': int(y.sum()),
        'roc_auc': float(roc_auc_score(y, p, sample_weight=sample_weight)),
        'average_precision': float(average_precision_score(y, p, sample_weight=sample_weight)),
        'brier_score': float(brier_score_loss(y, p, sample_weight=sample_weight)),
    }
