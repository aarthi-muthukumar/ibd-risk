from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

UNKNOWN = "__UNKNOWN__"


@dataclass(frozen=True)
class FieldSpec:
    name: str
    label: str
    group: str
    kind: Literal["age", "categorical", "binary", "fixed"]
    help: str
    options: tuple[tuple[str, str], ...] = ()
    default: object = None


def _opts(mapping: dict[str, str], unknown: bool = True) -> tuple[tuple[str, str], ...]:
    values = [(value, label) for value, label in mapping.items()]
    if unknown:
        values.append((UNKNOWN, "Unknown / not entered"))
    return tuple(values)


ACUTE = {
    "DX_ANEMIA": "Anemia",
    "DX_DEHYDRATION": "Dehydration",
    "DX_ELECTROLYTE_DISORDER": "Electrolyte disorder",
    "DX_GASTROINTESTINAL_BLEEDING": "Gastrointestinal bleeding",
    "DX_INTESTINAL_OBSTRUCTION": "Intestinal obstruction",
    "DX_SEPSIS": "Sepsis",
    "DX_VENOUS_THROMBOEMBOLISM": "Venous thromboembolism",
}

CMR = {
    "CMR_AIDS": "HIV/AIDS",
    "CMR_ALCOHOL": "Alcohol-related disorder",
    "CMR_ARTH": "Rheumatoid arthritis/collagen vascular disease",
    "CMR_CANCER_LEUK": "Leukemia",
    "CMR_CANCER_LYMPH": "Lymphoma",
    "CMR_CANCER_METS": "Metastatic cancer",
    "CMR_CANCER_NSITU": "Cancer in situ",
    "CMR_CANCER_SOLID": "Solid tumor without metastasis",
    "CMR_DEMENTIA": "Dementia",
    "CMR_DEPRESS": "Depression",
    "CMR_DIAB_CX": "Diabetes with complications",
    "CMR_DIAB_UNCX": "Diabetes without complications",
    "CMR_DRUG_ABUSE": "Drug-use disorder",
    "CMR_HTN_CX": "Hypertension with complications",
    "CMR_HTN_UNCX": "Hypertension without complications",
    "CMR_LUNG_CHRONIC": "Chronic pulmonary disease",
    "CMR_OBESE": "Obesity",
    "CMR_PERIVASC": "Peripheral vascular disease",
    "CMR_THYROID_HYPO": "Hypothyroidism",
    "CMR_THYROID_OTH": "Other thyroid disorder",
}


FIELDS: tuple[FieldSpec, ...] = (
    FieldSpec("AGE", "Age", "Patient characteristics", "age", "Age in years at ED admission; NEDS top-codes age 90 or older as 90.", default=54),
    FieldSpec("FEMALE", "Administrative sex", "Patient characteristics", "categorical", "NEDS binary administrative sex field; other or unavailable values are treated as unknown.", _opts({"0": "Male", "1": "Female"}), "0"),
    FieldSpec("RACE", "Administrative race or ethnicity", "Patient characteristics", "categorical", "NEDS combines race and ethnicity into one administrative field; ethnicity takes precedence when separately reported.", _opts({"1": "White", "2": "Black", "3": "Hispanic", "4": "Asian or Pacific Islander", "5": "Native American", "6": "Other"}), UNKNOWN),
    FieldSpec("ZIPINC_QRTL", "Area-level income category", "Social and access context", "categorical", "Area-level median household income category derived from residential ZIP code; this is not individual income.", _opts({"1": "Quartile 1 of 4", "2": "Quartile 2 of 4", "3": "Quartile 3 of 4", "4": "Quartile 4 of 4"}), UNKNOWN),
    FieldSpec("PAY1", "Primary expected payer", "Social and access context", "categorical", "Expected primary payer recorded in the administrative encounter.", _opts({"1": "Medicare", "2": "Medicaid", "3": "Private insurance", "4": "Self-pay", "5": "No charge", "6": "Other"}), UNKNOWN),
    FieldSpec("PL_NCHS", "Patient residence category", "Social and access context", "categorical", "NCHS urban–rural category for the patient's county of residence.", _opts({"1": "Large central metropolitan", "2": "Large fringe metropolitan", "3": "Medium metropolitan", "4": "Small metropolitan", "5": "Micropolitan", "6": "Noncore"}), UNKNOWN),
    FieldSpec("AWEEKEND", "Weekend encounter", "Social and access context", "categorical", "Whether the encounter began on Saturday or Sunday.", _opts({"0": "No", "1": "Yes"}), "0"),
    FieldSpec("YEAR", "Development year", "Hospital context", "fixed", "The fitted model contains only 2019 data; this value cannot be changed.", (("2019", "2019"),), "2019"),
    FieldSpec("HOSP_REGION", "Hospital region", "Hospital context", "categorical", "U.S. Census region of the hospital.", _opts({"1": "Northeast", "2": "Midwest", "3": "South", "4": "West"}), "3"),
    FieldSpec("HOSP_UR_TEACH", "Hospital teaching/location status", "Hospital context", "categorical", "HCUP teaching and metropolitan status.", _opts({"0": "Metropolitan, non-teaching", "1": "Metropolitan, teaching", "2": "Non-metropolitan"}), "1"),
    FieldSpec("HOSP_URCAT4", "Hospital urban/rural category", "Hospital context", "categorical", "HCUP hospital county urban–rural designation. Codes 7–9 were present in the fitted artifact but lack a reliable label in the reviewed uniform documentation.", _opts({"1": "Large metropolitan (≥1 million)", "2": "Small metropolitan (<1 million)", "3": "Micropolitan", "4": "Non-metropolitan/non-micropolitan", "7": "HCUP code 7 (collapsed/other)", "8": "HCUP code 8 (collapsed/other)", "9": "HCUP code 9 (collapsed/other)"}), "1"),
    FieldSpec("HOSP_TRAUMA", "Hospital trauma status", "Hospital context", "categorical", "Hospital trauma-center designation; some HCUP values are collapsed categories.", _opts({"0": "Not a trauma center", "1": "Level I", "2": "Level II", "3": "Level III", "7": "Level II or III (collapsed)", "8": "Level I or II (collapsed)"}), "0"),
    FieldSpec("HOSP_CONTROL", "Hospital ownership", "Hospital context", "categorical", "AHA-derived hospital ownership/control category.", _opts({"0": "Government or private (collapsed)", "1": "Government, nonfederal", "2": "Private, not-for-profit", "3": "Private, investor-owned", "4": "Private (collapsed)"}), "2"),
    *tuple(FieldSpec(name, label, "Current coded presentation", "binary", "Diagnosis flag derived from any ICD-10-CM position on the completed administrative encounter.", default=0) for name, label in ACUTE.items()),
    *tuple(FieldSpec(name, label, "Known medical history", "binary", "HCUP Elixhauser Comorbidity Software Refined indicator derived from diagnosis coding.", default=0) for name, label in CMR.items()),
)

SCHEMA_BY_NAME = {field.name: field for field in FIELDS}
APP_FEATURES = [field.name for field in FIELDS]
LEAKAGE_NAMES = {"EDevent", "DISP_ED", "DISP_IP", "DIED_VISIT", "LOS_IP", "TOTCHG_ED", "TOTCHG_IP", "KEY_ED", "DISCWT", "NEDS_STRATUM", "HOSPITAL_ADMISSION_FROM_ED", "HIGH_ACUITY_DISPOSITION"}


def build_input_frame(values: dict[str, object]) -> pd.DataFrame:
    unknown = sorted(set(values) - set(APP_FEATURES))
    if unknown:
        raise ValueError(f"Unsupported input fields: {', '.join(unknown)}")
    row: dict[str, object] = {}
    for field in FIELDS:
        value = values.get(field.name, field.default)
        if field.kind == "age":
            row[field.name] = np.nan if value in (None, UNKNOWN) else float(value)
        elif field.kind == "binary":
            if value not in (0, 1, False, True):
                raise ValueError(f"{field.label} must be yes or no.")
            row[field.name] = int(value)
        else:
            allowed = {option[0] for option in field.options}
            if value not in allowed:
                raise ValueError(f"Unsupported category for {field.label}.")
            row[field.name] = np.nan if value == UNKNOWN else str(value)
    return pd.DataFrame([row], columns=APP_FEATURES)
