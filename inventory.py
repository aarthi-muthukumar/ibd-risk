from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd


def inventory_hcup(paths: Optional[dict] = None) -> pd.DataFrame:
    data_dir = Path(paths['data_dir']) if paths else Path('NEDS_2019')
    rows = []
    for path in sorted(data_dir.rglob('*')):
        if path.is_file():
            rows.append({'path': str(path), 'size_bytes': path.stat().st_size})
    return pd.DataFrame(rows)
