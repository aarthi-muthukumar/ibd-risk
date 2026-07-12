from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

LOGGER = logging.getLogger(__name__)


def train_models(X: pd.DataFrame, y: pd.Series, sample_weight: Optional[pd.Series] = None):
    return train_logistic_regression(X, y, sample_weight=sample_weight)


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_features = X.select_dtypes(include=['number']).columns.tolist()
    categorical_features = X.select_dtypes(exclude=['number']).columns.tolist()

    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
    ])
    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
        ('onehot', OneHotEncoder(handle_unknown='ignore')),
    ])

    return ColumnTransformer([
        ('numeric', numeric_transformer, numeric_features),
        ('categorical', categorical_transformer, categorical_features),
    ])


def train_logistic_regression(X: pd.DataFrame, y: pd.Series, sample_weight: Optional[pd.Series] = None):
    preprocessor = build_preprocessor(X)
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(C=1.0, solver='lbfgs', random_state=42, max_iter=2000)),
    ])
    if sample_weight is not None:
        pipeline.fit(X, y, classifier__sample_weight=sample_weight)
    else:
        pipeline.fit(X, y)
    return pipeline
