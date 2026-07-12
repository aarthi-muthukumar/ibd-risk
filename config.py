from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml

ROOT = Path(__file__).resolve().parents[2]

DEFAULT_PATHS = {
    'data_dir': str(ROOT / 'NEDS_2019'),
    'reports_dir': str(ROOT / 'outputs'),
    'models_dir': str(ROOT / 'models'),
    'config_dir': str(ROOT / 'config'),
}


def load_paths(path: Optional[str] = None) -> dict[str, str]:
    if path is None:
        return DEFAULT_PATHS
    with open(path, 'r', encoding='utf-8') as handle:
        return yaml.safe_load(handle)
