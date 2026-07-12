from __future__ import annotations


import pandas as pd


def survey_design(weight='DISCWT', stratum='NEDS_STRATUM', cluster='HOSP_ED'):
    return {'weight': weight, 'stratum': stratum, 'cluster': cluster}


def weighted_rate(df: pd.DataFrame, outcome: str, weight: str = 'DISCWT') -> float:
    valid = df[[outcome, weight]].apply(pd.to_numeric, errors='coerce').dropna()
    if valid.empty or valid[weight].sum() <= 0:
        return float('nan')
    return float((valid[outcome] * valid[weight]).sum() / valid[weight].sum())
