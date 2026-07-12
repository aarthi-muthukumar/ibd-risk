from __future__ import annotations

from pathlib import Path
from typing import Iterator
import csv

import pandas as pd

CORE_2019_COLUMNS = [
    "AGE", "AMONTH", "AWEEKEND", "DIED_VISIT", "DISCWT", "DISP_ED", "DQTR",
    "EDevent", "FEMALE", "HCUPFILE", "HOSP_ED",
    *[f"I10_DX{i}" for i in range(1, 36)],
    "I10_INJURY", "I10_INJURY_CUT", "I10_INJURY_DROWN", "I10_INJURY_FALL",
    "I10_INJURY_FIRE", "I10_INJURY_FIREARM", "I10_INJURY_MACHINERY",
    "I10_INJURY_MVT", "I10_INJURY_NATURE", "I10_INJURY_OVEREXERT",
    "I10_INJURY_POISON", "I10_INJURY_STRUCK", "I10_INJURY_SUFFOCATION",
    "I10_INTENT_ASSAULT", "I10_INTENT_SELF_HARM", "I10_INTENT_UNINTENTIONAL",
    "I10_MULTINJURY", "I10_NDX", "KEY_ED", "NEDS_STRATUM", "PAY1", "PAY2",
    "PL_NCHS", "RACE", "TOTCHG_ED", "YEAR", "ZIPINC_QRTL",
]
HOSPITAL_2019_COLUMNS = [
    "DISCWT", "HOSPWT", "HOSP_CONTROL", "HOSP_ED", "HOSP_REGION", "HOSP_TRAUMA",
    "HOSP_URCAT4", "HOSP_UR_TEACH", "NEDS_STRATUM", "N_DISC_U", "N_HOSP_U",
    "S_DISC_U", "S_HOSP_U", "TOTAL_EDVisits", "YEAR",
]
CMR_2019_COLUMNS = [
    'CMR_AIDS', 'CMR_ALCOHOL', 'CMR_ARTH', 'CMR_CANCER_LEUK', 'CMR_CANCER_LYMPH',
    'CMR_CANCER_METS', 'CMR_CANCER_NSITU', 'CMR_CANCER_SOLID', 'CMR_DEMENTIA',
    'CMR_DEPRESS', 'CMR_DIAB_CX', 'CMR_DIAB_UNCX', 'CMR_DRUG_ABUSE', 'CMR_HTN_CX',
    'CMR_HTN_UNCX', 'CMR_LUNG_CHRONIC', 'CMR_OBESE', 'CMR_PERIVASC',
    'CMR_THYROID_HYPO', 'CMR_THYROID_OTH',
]


def read_headerless_csv(path: str | Path, columns: list[str], **kwargs):
    return pd.read_csv(path, header=None, names=columns, dtype=str, **kwargs)


def iter_core(path: str | Path, chunksize: int = 100_000, usecols: list[str] | None = None) -> Iterator[pd.DataFrame]:
    try:
        yield from read_headerless_csv(path, CORE_2019_COLUMNS, chunksize=chunksize, usecols=usecols)
    except NotImplementedError as exc:
        if 'encryption' in str(exc).lower():
            raise RuntimeError(
                'The HCUP Core archive is strongly encrypted. Extract NEDS_2019_CORE.csv '
                'with the HCUP-provided password, then pass that CSV via --core. Source files are never overwritten.'
            ) from exc
        raise


def read_hospitals(path: str | Path) -> pd.DataFrame:
    return read_headerless_csv(path, HOSPITAL_2019_COLUMNS)


def iter_core_uc(path: str | Path, batch_size: int = 25_000) -> Iterator[pd.DataFrame]:
    """Stream Core lines and parse only candidate K51 records; no rows are written."""
    batch: list[list[str]] = []
    with Path(path).open('r', encoding='utf-8', newline='') as handle:
        for line in handle:
            if 'K51' not in line.upper():
                continue
            row = next(csv.reader([line]))
            if len(row) != len(CORE_2019_COLUMNS):
                raise ValueError(f'Core row has {len(row)} fields; expected {len(CORE_2019_COLUMNS)}.')
            batch.append(row)
            if len(batch) >= batch_size:
                yield pd.DataFrame(batch, columns=CORE_2019_COLUMNS)
                batch = []
    if batch:
        yield pd.DataFrame(batch, columns=CORE_2019_COLUMNS)


def read_cmr_for_keys(path: str | Path, keys: set[str]) -> pd.DataFrame:
    """Read only Elixhauser CMR fields for selected encounter keys from the wide groups file."""
    rows: list[list[str]] = []
    with Path(path).open('r', encoding='utf-8', newline='') as handle:
        for line in handle:
            key = line.rsplit(',', 1)[-1].strip().strip('"\r\n')
            if key not in keys:
                continue
            leading = next(csv.reader([line], strict=True))
            rows.append([*leading[:20], key])
    return pd.DataFrame(rows, columns=[*CMR_2019_COLUMNS, 'KEY_ED'])
