from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from .model_schema import APP_FEATURES


class ArtifactError(RuntimeError):
    pass


def load_model_bundle(path: str | Path) -> dict:
    path = Path(path)
    if not path.is_file():
        raise ArtifactError(f"Model artifact is missing: {path}")
    with path.open("rb") as handle:
        bundle = pickle.load(handle)
    if not isinstance(bundle, dict) or not {"model", "features", "endpoint"}.issubset(bundle):
        raise ArtifactError("Model artifact has an unsupported structure.")
    model = bundle["model"]
    if not isinstance(model, Pipeline) or not hasattr(model, "predict_proba"):
        raise ArtifactError("The artifact does not contain a fitted probability pipeline.")
    expected = list(model.feature_names_in_)
    if list(bundle["features"]) != expected or expected != APP_FEATURES:
        raise ArtifactError("Application schema does not match the fitted model feature order.")
    return bundle


def load_metadata(path: str | Path) -> dict:
    path = Path(path)
    if not path.is_file():
        raise ArtifactError(f"Model metadata is missing: {path}")
    with path.open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    required = {"endpoint", "cohort_size", "event_count", "included_predictors", "model_artifact_filename"}
    if not required.issubset(metadata):
        raise ArtifactError("Model metadata is incomplete.")
    return metadata


def admission_class_index(model: Pipeline, admission_class: int = 1) -> int:
    classes = list(model.named_steps["classifier"].classes_)
    if admission_class not in classes:
        raise ArtifactError(f"Admission class {admission_class!r} is absent from classifier classes {classes!r}.")
    return classes.index(admission_class)


def predict_admission_probability(bundle: dict, input_df: pd.DataFrame) -> float:
    model = bundle["model"]
    probabilities = np.asarray(model.predict_proba(input_df))
    index = admission_class_index(model)
    if probabilities.shape != (1, len(model.named_steps["classifier"].classes_)):
        raise ArtifactError("Prediction returned an unexpected shape.")
    probability = float(probabilities[0, index])
    if not np.isfinite(probability) or not 0 <= probability <= 1:
        raise ArtifactError("Prediction did not return a valid probability.")
    return probability


def descriptive_band(probability: float, metadata: dict) -> str:
    bands = metadata["descriptive_probability_bands"]
    if probability < bands["lower_upper_bound"]:
        return "Lower observed admission probability"
    if probability < bands["intermediate_upper_bound"]:
        return "Intermediate observed admission probability"
    return "Higher observed admission probability"

