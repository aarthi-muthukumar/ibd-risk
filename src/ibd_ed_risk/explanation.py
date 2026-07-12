from __future__ import annotations

import numpy as np
import pandas as pd


def logistic_contributions(model, input_df: pd.DataFrame) -> dict:
    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]
    if classifier.coef_.shape[0] != 1:
        raise ValueError("Only binary logistic-regression explanations are supported.")
    transformed = preprocessor.transform(input_df)
    vector = transformed.toarray()[0] if hasattr(transformed, "toarray") else np.asarray(transformed)[0]
    names = preprocessor.get_feature_names_out()
    coefficients = classifier.coef_[0]
    contributions = vector * coefficients
    intercept = float(classifier.intercept_[0])
    decision = float(model.decision_function(input_df)[0])
    reconstructed = intercept + float(contributions.sum())
    if not np.isclose(decision, reconstructed, rtol=1e-9, atol=1e-9):
        raise ValueError("Coefficient contributions do not reconstruct the model decision function.")
    rows = [
        {
            "transformed_feature": str(name),
            "transformed_value": float(value),
            "coefficient": float(coefficient),
            "log_odds_contribution": float(contribution),
            "direction": "increased" if contribution > 0 else "decreased" if contribution < 0 else "neutral",
        }
        for name, value, coefficient, contribution in zip(names, vector, coefficients, contributions)
        if not np.isclose(contribution, 0.0)
    ]
    rows.sort(key=lambda row: abs(row["log_odds_contribution"]), reverse=True)
    return {"intercept": intercept, "decision_function": decision, "reconstructed": reconstructed, "contributions": rows}
