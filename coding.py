from __future__ import annotations

import re
from collections.abc import Iterable, Mapping


def normalize_diagnosis(code: object) -> str | None:
    """Normalize an ICD-10-CM value for prefix matching without guessing invalid codes."""
    if code is None:
        return None
    value = re.sub(r"[^A-Z0-9]", "", str(code).strip().upper())
    return value or None


def has_prefix(code: object, prefixes: Iterable[str]) -> bool:
    value = normalize_diagnosis(code)
    normalized = tuple(p for p in (normalize_diagnosis(x) for x in prefixes) if p)
    return bool(value and value.startswith(normalized))


def diagnosis_flags(values: Iterable[object], definitions: Mapping[str, Iterable[str]]) -> dict[str, int]:
    normalized = [normalize_diagnosis(value) for value in values]
    return {
        name: int(any(code and code.startswith(tuple(prefixes)) for code in normalized))
        for name, prefixes in definitions.items()
    }
