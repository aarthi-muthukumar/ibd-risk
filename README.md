# HCUP-NEDS 2019 UC ED Admission Pipeline

This repository contains a conservative Python pipeline for HCUP-NEDS 2019. The uncompressed ED CSV is a supplemental procedure file; cohort construction must use `NEDS_2019_CORE.zip` and hospital characteristics come from `NEDS_2019_HOSPITAL.zip`.

## What this pipeline does
- Inventories the local HCUP files without reading row-level values.
- Builds an ulcerative-colitis-focused cohort by scanning all available diagnosis columns for ICD-10-CM K51 codes.
- Joins hospital characteristics and documented Elixhauser Refined comorbidity indicators.
- Performs survey-weighted model fitting with five-fold hospital-grouped internal validation.
- Writes aggregate reports only; encounter-level HCUP rows and predictions are not exported.

## UC ICD-10 definition
- UC-related encounters are defined using ICD-10-CM diagnosis code K51 and all K51 subcodes.
- The pipeline scans every available I10_DX column for a match.
- Two definitions are supported:
  - --uc-definition any: any diagnosis position contains K51
  - --uc-definition primary: the first diagnosis position contains K51

## Outcome definition
The primary outcome is `EDevent == 2` (admitted to the same hospital). The secondary high-acuity endpoint is `EDevent` in `{2, 3, 9}`. These codes come from the official HCUP NEDS data-element documentation; missing or undocumented values are excluded.

## Why this does not predict a 12-month repeat ED visit
NEDS is encounter-level and does not contain a longitudinal patient identifier sufficient to track prior or future ED use across years. The model is therefore framed as a prediction of whether the index ED encounter results in hospital admission or another high-acuity disposition.

## Why logistic regression is used
The outcome is binary. Logistic regression is a transparent, interpretable baseline model suitable for this setting and is used as the primary model. Ordinary linear regression is not appropriate for a binary outcome.

## Leakage precautions
The pipeline avoids using post-disposition variables such as DISP_ED, DISP_IP, DIED_VISIT, LOS_IP, or inpatient procedure variables as predictors. It also separates descriptive-only, survey-design, and identifier columns from the model features.

## Survey-weight limitations
The scripts preserve DISCWT and NEDS_STRATUM for descriptive analyses, but the provided scikit-learn evaluation is weighted only for model fitting and is not a complete complex-survey variance analysis.

## Clinical use disclaimer
This calculator is a workflow prototype and requires external validation before any clinical use.

## Running the clinician-facing research prototype

Model training is already complete. Launching the app uses only the fitted model and aggregate metadata; it does not require, open, or scan the source NEDS files.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

The required fitted artifact is `models/final_model.joblib`; app metadata is stored in `models/model_metadata.json`.

The app estimates the probability that a UC-related ED encounter is admitted to the same hospital. It is an investigational research prototype, does not measure biological UC severity, and must not be used to recommend admission, discharge, or treatment.

The patient residential ZIP entry is resolved entirely offline using `data_reference/zip_income_quartile_2019.csv`. The lookup contains public 2019 ACS 5-year ZCTA B19013 median household income estimates and derived categories based on official 2019 HCUP thresholds. The entered ZIP is used only in memory and is not stored, logged, transmitted, or passed to the fitted model. Because NEDS used Claritas ZIP estimates rather than ACS ZCTA estimates, the derived category is an approximation of the original `ZIPINC_QRTL` variable.

## Scripts
- python scripts/inventory_hcup.py
- Extract `NEDS_2019_CORE.csv` locally using the HCUP-provided archive password.
- python scripts/build_uc_cohort.py --core NEDS_2019/NEDS_2019_CORE.csv --report-dir reports
- python scripts/train_admission_model.py --core NEDS_2019/NEDS_2019_CORE.csv --hospital NEDS_2019/NEDS_2019_HOSPITAL.csv --diagnosis-groups NEDS_2019/NEDS_2019_DX_PR_GRPS.csv
- python scripts/train_uc_admission_model.py --input outputs/uc_neds_2019_modeling.csv --output-dir outputs --model-dir models
- python scripts/evaluate_uc_admission_model.py --input outputs/uc_neds_2019_modeling.csv --model models/final_model.joblib --output-dir outputs
- python scripts/generate_risk_outputs.py --input outputs/uc_neds_2019_modeling.csv --model models/final_model.joblib --output-dir outputs
- streamlit run app.py

## Output files
- reports/hcup_file_inventory.csv
- reports/cohort_summary.csv
- reports/model_feature_list.csv
- reports/model_performance.csv
- models/final_model.joblib

Row-level source data, modeling datasets, and predictions are intentionally excluded from outputs and Git.
