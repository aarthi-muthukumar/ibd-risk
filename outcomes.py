from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

LOGGER = logging.getLogger(__name__)


def inspect_outcome_variables(df: pd.DataFrame, output_path: str | Path) -> pd.DataFrame:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    review_rows = []
    for col in ['EDevent', 'DISP_ED', 'DISP_IP']:
        if col in df.columns:
            counts = df[col].value_counts(dropna=False)
            review_rows.append(pd.DataFrame({'variable': col, 'value': counts.index, 'count': counts.values}))
    if not review_rows:
        review_rows = [pd.DataFrame({'variable': [], 'value': [], 'count': []})]
    review = pd.concat(review_rows, ignore_index=True) if review_rows else pd.DataFrame(columns=['variable', 'value', 'count'])
    review['percent'] = review['count'] / len(df) if len(df) else None
    review['human_review_label'] = ''
    review.to_csv(output_path, index=False)
    return review


def derive_outcomes(df: pd.DataFrame) -> pd.DataFrame:
    """Derive documented NEDS endpoints from EDevent (HCUP uniform values)."""
    if 'EDevent' not in df.columns:
        raise ValueError('Outcome cannot be derived because EDevent is unavailable.')
    outcome_df = df.copy()
    event = pd.to_numeric(outcome_df['EDevent'], errors='coerce')
    known = event.isin([1, 2, 3, 9, 98, 99])
    outcome_df['HOSPITAL_ADMISSION_FROM_ED'] = event.eq(2).where(known).astype('Int64')
    outcome_df['ED_DEATH'] = event.eq(9).where(known).astype('Int64')
    outcome_df['HIGH_ACUITY_DISPOSITION'] = event.isin([2, 3, 9]).where(known).astype('Int64')
    outcome_df['OUTCOME_REVIEW_REQUIRED'] = ~known
    outcome_df['OUTCOME_REVIEW_REASON'] = (~known).map({True: 'Missing or undocumented EDevent value.', False: ''})
    return outcome_df
