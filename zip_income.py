from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal

HCUP_2019_THRESHOLDS = (48_000, 61_000, 82_000)
DEFAULT_LOOKUP_PATH = Path(__file__).resolve().parents[2] / "data_reference" / "zip_income_quartile_2019.csv"


class InvalidZipCode(ValueError):
    pass


@dataclass(frozen=True)
class ZipIncomeResult:
    status: Literal["available", "unavailable"]
    quartile: int | None = None
    median_household_income: int | None = None


def income_to_quartile(income: int) -> int:
    if not isinstance(income, int) or isinstance(income, bool) or income < 0:
        raise ValueError("Median household income must be a nonnegative integer.")
    lower_q2, lower_q3, lower_q4 = HCUP_2019_THRESHOLDS
    if income < lower_q2:
        return 1
    if income < lower_q3:
        return 2
    if income < lower_q4:
        return 3
    return 4


def validate_zip_code(zip_code: str) -> str:
    if not isinstance(zip_code, str) or not re.fullmatch(r"\d{5}", zip_code):
        raise InvalidZipCode("Patient residential ZIP code must contain exactly five digits.")
    return zip_code


@lru_cache(maxsize=4)
def load_zip_income_lookup(path: str | Path = DEFAULT_LOOKUP_PATH) -> dict[str, tuple[int, int]]:
    lookup_path = Path(path)
    if not lookup_path.is_file():
        raise FileNotFoundError(f"Local ZIP-income lookup is missing: {lookup_path}")
    lookup: dict[str, tuple[int, int]] = {}
    with lookup_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        expected = {"zip_code", "median_household_income_acs2019", "derived_income_quartile"}
        if set(reader.fieldnames or ()) != expected:
            raise ValueError("ZIP-income lookup has an unexpected schema.")
        for row in reader:
            zip_code = validate_zip_code(row["zip_code"])
            income = int(row["median_household_income_acs2019"])
            quartile = int(row["derived_income_quartile"])
            if quartile != income_to_quartile(income):
                raise ValueError(f"ZIP-income lookup quartile is inconsistent for {zip_code}.")
            lookup[zip_code] = (income, quartile)
    return lookup


def lookup_zip_income(zip_code: str, path: str | Path = DEFAULT_LOOKUP_PATH) -> ZipIncomeResult:
    normalized = validate_zip_code(zip_code)
    match = load_zip_income_lookup(path).get(normalized)
    if match is None:
        return ZipIncomeResult(status="unavailable")
    income, quartile = match
    return ZipIncomeResult(status="available", quartile=quartile, median_household_income=income)
