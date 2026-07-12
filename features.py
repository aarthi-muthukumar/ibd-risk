from __future__ import annotations

import pandas as pd

OUTCOME_COLUMNS = {
    'EDevent', 'DISP_ED', 'DISP_IP', 'DIED_VISIT', 'HOSPITAL_ADMISSION_FROM_ED',
    'HIGH_ACUITY_DISPOSITION', 'ED_DEATH', 'OUTCOME_REVIEW_REQUIRED', 'OUTCOME_REVIEW_REASON',
}
POST_DISPOSITION_COLUMNS = {'LOS_IP', 'TOTCHG_ED', 'TOTCHG_IP'}
IDENTIFIER_COLUMNS = {'KEY_ED'}
SURVEY_ONLY_COLUMNS = {'DISCWT', 'NEDS_STRATUM'}
DEFAULT_EXCLUDED_PREFIXES = ('APRDRG_', 'APR_', 'I10_PR_IP', 'PR_IP')


def leakage_columns(columns) -> list[str]:
    excluded = OUTCOME_COLUMNS | POST_DISPOSITION_COLUMNS | IDENTIFIER_COLUMNS | SURVEY_ONLY_COLUMNS
    return sorted(c for c in columns if c in excluded or c.startswith(DEFAULT_EXCLUDED_PREFIXES))


def construct_features(df: pd.DataFrame, manifest: pd.DataFrame | None = None) -> pd.DataFrame:
    if manifest is not None:
        allowed = manifest.loc[manifest['include'].astype(str).str.lower().eq('true'), 'variable']
        return df[[c for c in allowed if c in df.columns]].copy()
    return df.drop(columns=leakage_columns(df.columns), errors='ignore').copy()
