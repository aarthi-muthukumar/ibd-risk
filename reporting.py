from __future__ import annotations

from pathlib import Path

import pandas as pd


def suppress_small_cells(table: pd.DataFrame, count_column: str = 'n', threshold: int = 11) -> pd.DataFrame:
    result = table.copy()
    if count_column in result:
        mask = pd.to_numeric(result[count_column], errors='coerce').lt(threshold)
        result.loc[mask, count_column] = pd.NA
        for column in result.columns:
            if column != count_column and ('rate' in column.lower() or 'percent' in column.lower()):
                result.loc[mask, column] = pd.NA
    return result


def write_reports(tables: dict[str, pd.DataFrame], report_dir: str | Path) -> None:
    destination = Path(report_dir)
    destination.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        suppress_small_cells(table).to_csv(destination / f'{name}.csv', index=False)
