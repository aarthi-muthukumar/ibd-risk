from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd

from .cohort import build_uc_cohort, load_ed_core
from .column_inventory import build_column_inventory
from .outcomes import derive_outcomes, inspect_outcome_variables

LOGGER = logging.getLogger(__name__)


def build_full_pipeline(input_path: str | Path, output_dir: str | Path, uc_definition: str = 'any', chunksize: Optional[int] = None) -> dict:
    raise RuntimeError(
        'Legacy row-export pipeline disabled to protect HCUP data. '
        'Use scripts/build_uc_cohort.py with the NEDS Core archive for aggregate reports.'
    )
    # Historical implementation retained below for reference; it is intentionally unreachable.
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info('Loading NEDS CSV from %s', input_path)
    if chunksize is not None:
        chunks = load_ed_core(input_path, chunksize=chunksize, low_memory=False)
        first_chunk = None
        for chunk in chunks:
            first_chunk = chunk if first_chunk is None else pd.concat([first_chunk, chunk], ignore_index=True)
            break
        df = first_chunk if first_chunk is not None else pd.DataFrame()
    else:
        df = load_ed_core(input_path, low_memory=False)

    inventory_df = build_column_inventory(df, output_dir / 'column_inventory.csv')
    inventory_df.to_csv(output_dir / 'column_inventory.csv', index=False)

    LOGGER.info('Building UC cohort with %s definition', uc_definition)
    cohort = build_uc_cohort(input_path, uc_definition=uc_definition)

    LOGGER.info('Inspecting outcome variables')
    outcome_review = inspect_outcome_variables(cohort, output_dir / 'outcome_coding_review.csv')
    outcome_review.to_csv(output_dir / 'outcome_coding_review.csv', index=False)

    outcome_df = derive_outcomes(cohort)
    filtered = outcome_df.dropna(subset=['HOSPITAL_ADMISSION_FROM_ED'])

    filtered.to_csv(output_dir / 'uc_neds_2019_filtered.csv', index=False)
    cohort.to_csv(output_dir / 'uc_neds_2019_modeling.csv', index=False)
    cohort.to_csv(output_dir / 'uc_neds_2019_diagnoses.csv', index=False)

    cohort_flow = pd.DataFrame([
        {'metric': 'total_neds_encounters', 'value': int(len(df))},
        {'metric': 'encounters_with_any_k51', 'value': int(cohort['UC_ANY_DX'].sum())},
        {'metric': 'encounters_with_primary_k51', 'value': int(cohort['UC_PRIMARY_DX'].sum())},
        {'metric': 'excluded_for_missing_or_invalid_outcome', 'value': int(len(cohort) - len(filtered))},
        {'metric': 'final_model_sample_size', 'value': int(len(filtered))},
    ])
    cohort_flow.to_csv(output_dir / 'cohort_flow.csv', index=False)
    missingness = filtered.isna().mean().reset_index()
    missingness.columns = ['column', 'percent_missing']
    missingness.to_csv(output_dir / 'missingness_summary.csv', index=False)

    return {
        'column_inventory': inventory_df,
        'cohort': cohort,
        'outcome_review': outcome_review,
        'filtered': filtered,
        'output_dir': str(output_dir),
    }
