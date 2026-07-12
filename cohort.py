from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import pandas as pd

from .coding import normalize_diagnosis


def normalize_dx(code: object) -> Optional[str]:
    if pd.isna(code):
        return None
    return normalize_diagnosis(code)


def load_ed_core(ed_path: str | Path, nrows: Optional[int] = None, chunksize: Optional[int] = None, low_memory: bool = False):
    path = Path(ed_path)
    if not path.exists():
        raise FileNotFoundError(f'Input file not found: {path}')
    if chunksize is not None:
        return pd.read_csv(path, header=None, chunksize=chunksize, low_memory=low_memory, dtype=str)

    cols = [f'V{i}' for i in range(73)]
    return pd.read_csv(path, header=None, names=cols, nrows=nrows, low_memory=low_memory, dtype=str)


def infer_diagnosis_columns(df: pd.DataFrame) -> list[str]:
    return sorted(
        [col for col in df.columns if isinstance(col, str) and col.startswith("I10_DX")],
        key=lambda name: int(name.removeprefix("I10_DX")),
    )


def annotate_uc_cohort(df: pd.DataFrame, diagnosis_columns: Optional[Sequence[str]] = None, primary_column: str = 'I10_DX1') -> pd.DataFrame:
    if diagnosis_columns is None:
        diagnosis_columns = infer_diagnosis_columns(df)
    if not diagnosis_columns:
        raise ValueError('Unable to identify diagnosis columns to scan for UC codes.')

    cohort = df.copy()
    diagnoses = cohort[list(diagnosis_columns)].fillna('').apply(
        lambda column: column.str.strip().str.upper().str.replace('.', '', regex=False)
    )
    cohort[list(diagnosis_columns)] = diagnoses
    uc_matrix = diagnoses.apply(lambda column: column.str.startswith('K51'))
    any_dx = uc_matrix.any(axis=1)
    primary_dx = diagnoses[primary_column].str.startswith('K51')
    first_index = uc_matrix.to_numpy().argmax(axis=1) + 1
    dx_positions = pd.Series(first_index, index=cohort.index).where(any_dx)

    cohort['UC_ANY_DX'] = any_dx.astype(int)
    cohort['UC_PRIMARY_DX'] = primary_dx.astype(int)
    cohort['UC_DX_POSITION'] = dx_positions
    cohort['CROHNS_ANY_DX'] = diagnoses.apply(lambda column: column.str.startswith('K50')).any(axis=1).astype(int)
    cohort['K50_K51_OVERLAP'] = ((cohort['UC_ANY_DX'] == 1) & (cohort['CROHNS_ANY_DX'] == 1)).astype(int)
    return cohort


def build_uc_cohort(ed_path: str | Path, nrows: Optional[int] = None, uc_definition: str = 'any', diagnosis_columns: Optional[Sequence[str]] = None, primary_column: str = 'V2') -> pd.DataFrame:
    df = load_ed_core(ed_path, nrows=nrows)
    cohort = annotate_uc_cohort(df, diagnosis_columns=diagnosis_columns, primary_column=primary_column)
    if uc_definition == 'primary':
        return cohort.loc[cohort['UC_PRIMARY_DX'] == 1].copy()
    if uc_definition == 'any':
        return cohort.loc[cohort['UC_ANY_DX'] == 1].copy()
    raise ValueError("uc_definition must be 'any' or 'primary'")
