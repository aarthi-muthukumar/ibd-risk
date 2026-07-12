from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import pandas as pd

LOGGER = logging.getLogger(__name__)

REQUESTED_VARIABLES = [
    'AGE',
    'FEMALE',
    'RACE',
    'ZIPINC_QRTL',
    'PAY1',
    'AWEEKEND',
    'AMONTH',
    'DQTR',
    'EDevent',
    'DIED_VISIT',
    'DISP_ED',
    'DISP_IP',
    'TOTCHG_ED',
    'TOTCHG_IP',
    'LOS_IP',
    'I10_DX1',
    'I10_NDX',
    'I10_PR_IP1',
    'I10_NPR_IP',
    'HOSP_REGION',
    'HOSP_UR_TEACH',
    'HOSP_URCAT4',
    'PL_NCHS',
    'YEAR',
    'DISCWT',
    'NEDS_STRATUM',
    'HOSP_ED',
    'KEY_ED',
]


def build_column_inventory(df: pd.DataFrame, output_path: str | Path) -> pd.DataFrame:
    output_path = Path(output_path)
    columns = [col for col in df.columns]
    inventory_rows = []
    for variable in REQUESTED_VARIABLES:
        if variable in columns:
            matched_column = variable
            match_type = 'exact'
        else:
            matched_column = None
            match_type = 'unavailable'
            for col in columns:
                if isinstance(col, str) and col.startswith(variable):
                    matched_column = col
                    match_type = 'prefix_match'
                    break
            if matched_column is None:
                for col in columns:
                    if isinstance(col, str) and variable.lower() in col.lower():
                        matched_column = col
                        match_type = 'renamed_equivalent'
                        break
        if matched_column is None:
            values = None
            missing = None
            data_type = None
            notes = 'Column unavailable in input file.'
        else:
            series = df[matched_column]
            data_type = str(series.dtype)
            missing = series.isna().mean()
            values = series.dropna().astype(str).unique()[:10]
            notes = ''
        inventory_rows.append({
            'requested_variable': variable,
            'matched_column': matched_column,
            'match_type': match_type,
            'data_type': data_type,
            'percent_missing': missing,
            'unique_values': values,
            'notes': notes,
        })
    inventory_df = pd.DataFrame(inventory_rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    inventory_df.to_csv(output_path, index=False)
    return inventory_df
