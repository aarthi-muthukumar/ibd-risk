from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import sparse


def logistic_contributions(model, input_df: pd.DataFrame) -> dict:
    """Calculate per-feature logistic-regression contributions.

    The preprocessing pipeline is executed once. Sparse transformed inputs
    remain sparse rather than being converted into a dense matrix.
    """
    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]

    if classifier.coef_.shape[0] != 1:
        raise ValueError(
            "Only binary logistic-regression explanations are supported."
        )

    transformed = preprocessor.transform(input_df)

    if transformed.shape[0] != 1:
        raise ValueError(
            "Exactly one encounter is required for an individual explanation."
        )

    names = np.asarray(
        preprocessor.get_feature_names_out(),
        dtype=object,
    )
    coefficients = np.asarray(classifier.coef_[0], dtype=float)

    if transformed.shape[1] != coefficients.shape[0]:
        raise ValueError(
            "The transformed feature count does not match the classifier "
            "coefficient count."
        )

    intercept = float(classifier.intercept_[0])

    rows: list[dict] = []

    if sparse.issparse(transformed):
        row = transformed.getrow(0).tocsr()

        active_indices = row.indices
        active_values = row.data.astype(float, copy=False)
        active_coefficients = coefficients[active_indices]
        active_contributions = active_values * active_coefficients

        contribution_sum = float(active_contributions.sum())

        for index, value, coefficient, contribution in zip(
            active_indices,
            active_values,
            active_coefficients,
            active_contributions,
        ):
            contribution = float(contribution)

            if math.isclose(contribution, 0.0, abs_tol=1e-15):
                continue

            rows.append(
                {
                    "transformed_feature": str(names[index]),
                    "transformed_value": float(value),
                    "coefficient": float(coefficient),
                    "log_odds_contribution": contribution,
                    "direction": (
                        "increased"
                        if contribution > 0
                        else "decreased"
                    ),
                }
            )
    else:
        vector = np.asarray(transformed, dtype=float)[0]
        contributions = vector * coefficients
        contribution_sum = float(contributions.sum())

        for name, value, coefficient, contribution in zip(
            names,
            vector,
            coefficients,
            contributions,
        ):
            contribution = float(contribution)

            if math.isclose(contribution, 0.0, abs_tol=1e-15):
                continue

            rows.append(
                {
                    "transformed_feature": str(name),
                    "transformed_value": float(value),
                    "coefficient": float(coefficient),
                    "log_odds_contribution": contribution,
                    "direction": (
                        "increased"
                        if contribution > 0
                        else "decreased"
                    ),
                }
            )

    reconstructed = intercept + contribution_sum

    if not math.isfinite(reconstructed):
        raise ValueError(
            "The reconstructed model decision value is not finite."
        )

    rows.sort(
        key=lambda row: abs(row["log_odds_contribution"]),
        reverse=True,
    )

    return {
        "intercept": intercept,
        "decision_function": reconstructed,
        "reconstructed": reconstructed,
        "contributions": rows,
    }
