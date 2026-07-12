from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src.ibd_ed_risk.app_utils import (
    ArtifactError,
    descriptive_band,
    load_metadata,
    load_model_bundle,
    predict_admission_probability,
)
from src.ibd_ed_risk.explanation import logistic_contributions
from src.ibd_ed_risk.model_schema import FIELDS, SCHEMA_BY_NAME, UNKNOWN, build_input_frame
from src.ibd_ed_risk.zip_income import InvalidZipCode, lookup_zip_income

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "final_model.joblib"
METADATA_PATH = ROOT / "models" / "model_metadata.json"
PERFORMANCE_PATH = ROOT / "reports" / "model_performance.csv"

st.set_page_config(page_title="UC ED Admission Risk Research Prototype", page_icon="🔬", layout="wide")


@st.cache_resource
def get_artifacts():
    return load_model_bundle(MODEL_PATH), load_metadata(METADATA_PATH)


def option_label(field, value: str) -> str:
    return dict(field.options).get(value, value)


def render_input(field):
    if field.kind == "age":
        age_unknown = st.checkbox("Age unknown / not entered", value=False, key="AGE_unknown")
        age = st.number_input(field.label, min_value=0, max_value=90, value=int(field.default), step=1, help=field.help, disabled=age_unknown)
        return UNKNOWN if age_unknown else age
    if field.kind == "binary":
        return int(st.checkbox(field.label, value=bool(field.default), help=field.help))
    if field.kind == "fixed":
        st.text_input(field.label, value=str(field.default), disabled=True, help=field.help)
        return str(field.default)
    values = [value for value, _ in field.options]
    default_index = values.index(field.default) if field.default in values else 0
    return st.selectbox(
        field.label,
        values,
        index=default_index,
        format_func=lambda value: option_label(field, value),
        help=field.help,
    )


try:
    bundle, metadata = get_artifacts()
except Exception as exc:
    st.error(f"The research model could not be loaded. {exc}")
    st.stop()

st.title("UC ED Admission Risk Research Prototype")
st.warning(
    "Investigational research use only. This model estimates admission patterns observed in the 2019 "
    "Nationwide Emergency Department Sample and has not been externally or prospectively validated for "
    "bedside decision-making."
)
st.info("This tool estimates observed disposition patterns. It does not determine whether a patient should be admitted or discharged.")

calculator_tab, performance_tab, about_tab = st.tabs(["Risk estimate", "Model performance", "About the model"])

with calculator_tab:
    st.subheader("Encounter characteristics")
    st.caption("Enter administrative characteristics as they would be coded in NEDS. No entered values are stored or transmitted.")
    values: dict[str, object] = {}
    with st.form("admission-risk-form", clear_on_submit=False):
        patient_zip = ""
        zip_unavailable = False
        for group in ["Patient characteristics", "Social and access context", "Known medical history", "Current coded presentation", "Hospital context"]:
            with st.expander(group, expanded=group in {"Patient characteristics", "Social and access context"}):
                group_fields = [field for field in FIELDS if field.group == group]
                columns = st.columns(2)
                for index, field in enumerate(group_fields):
                    with columns[index % 2]:
                        if field.name == "ZIPINC_QRTL":
                            zip_unavailable = st.checkbox("ZIP code unavailable", value=False)
                            patient_zip = st.text_input(
                                "Patient residential ZIP code",
                                value="",
                                max_chars=5,
                                disabled=zip_unavailable,
                                help="Used to estimate the median household income category of the patient’s residential area. This is not the patient’s individual income.",
                                placeholder="e.g., 02138",
                            )
                        else:
                            values[field.name] = render_input(field)
                if group == "Patient characteristics":
                    st.warning(
                        "Race in NEDS is an administrative variable and does not represent a biological risk factor. "
                        "Its inclusion may reflect structural and healthcare-system patterns. Model performance should "
                        "also be evaluated in future versions without race."
                    )
                if group in {"Known medical history", "Current coded presentation"}:
                    st.caption("These indicators reflect completed administrative diagnosis coding and may not be available in real time.")
        submitted = st.form_submit_button("Estimate admission probability", type="primary")

    if submitted:
        try:
            zip_result = None
            if zip_unavailable:
                values["ZIPINC_QRTL"] = UNKNOWN
            else:
                zip_result = lookup_zip_income(patient_zip)
                values["ZIPINC_QRTL"] = str(zip_result.quartile) if zip_result.status == "available" else UNKNOWN
            input_df = build_input_frame(values)
            probability = predict_admission_probability(bundle, input_df)
            explanation = logistic_contributions(bundle["model"], input_df)
        except (ValueError, InvalidZipCode, ArtifactError) as exc:
            st.error(f"An estimate could not be produced: {exc}")
        except Exception as exc:
            st.error(f"The model encountered an unexpected inference error: {exc}")
        else:
            if zip_unavailable:
                st.info("Area-level income category: unavailable; the fitted model's supported missing-value behavior was used.")
            elif zip_result and zip_result.status == "available":
                st.success(f"Area-level income category: Quartile {zip_result.quartile} of 4")
                st.caption("Derived locally from the 2019 ACS 5-year ZCTA median household income estimate using official 2019 HCUP dollar thresholds.")
            else:
                st.info("Area-level income category: unavailable for this ZIP code; the fitted model's supported missing-value behavior was used. No nearby ZIP was substituted.")
            prevalence = metadata["event_count"] / metadata["cohort_size"]
            st.divider()
            left, middle, right = st.columns(3)
            left.metric("Estimated admission probability", f"{probability:.1%}")
            middle.metric("Descriptive range", descriptive_band(probability, metadata))
            right.metric("Observed development prevalence", f"{prevalence:.1%}")
            st.write(
                f"Among NEDS encounters with similar coded characteristics, the estimated probability of admission "
                f"to the same hospital was **{probability:.1%}**."
            )
            st.caption(f"Overall observed admission prevalence: {metadata['event_count']:,} / {metadata['cohort_size']:,} ({prevalence:.1%}).")
            st.warning("These categories describe modeled admission frequency and are not admission or discharge thresholds.")

            with st.expander("Inputs used for this estimate"):
                summary = []
                row = input_df.iloc[0]
                for field in FIELDS:
                    raw = row[field.name]
                    if field.kind == "binary":
                        display = "Yes" if int(raw) else "No"
                    elif pd.isna(raw):
                        display = "Unknown / not entered"
                    elif field.options:
                        display = option_label(field, str(raw))
                    elif field.kind == "age":
                        display = str(int(raw))
                    else:
                        display = str(raw)
                    summary.append({"Field": field.label, "Entered value": display, "Model variable": field.name})
                st.dataframe(pd.DataFrame(summary), hide_index=True, use_container_width=True)

            with st.expander("How this estimate was calculated"):
                st.write(
                    "The fitted logistic regression combines a model intercept with transformed input contributions. "
                    "Contributions describe statistical associations in the NEDS development data; they are not causal "
                    "effects or reasons to admit a patient."
                )
                st.metric("Model intercept (log-odds)", f"{explanation['intercept']:.4f}")
                contribution_table = pd.DataFrame(explanation["contributions"])
                if not contribution_table.empty:
                    contribution_table["Interpretation"] = contribution_table["direction"].map({
                        "increased": "Associated with higher modeled admission probability",
                        "decreased": "Associated with lower modeled admission probability",
                        "neutral": "No change in modeled probability",
                    })
                    st.dataframe(
                        contribution_table[["transformed_feature", "coefficient", "log_odds_contribution", "Interpretation"]],
                        hide_index=True,
                        use_container_width=True,
                    )
                st.caption(
                    f"Intercept plus all displayed contributions reconstructs the model decision value: "
                    f"{explanation['reconstructed']:.6f}."
                )

with performance_tab:
    st.header("Model performance")
    main = {
        "Development encounters": f"{metadata['cohort_size']:,}",
        "Admission events": f"{metadata['event_count']:,}",
        "ROC AUC": f"{metadata['auroc']:.4f}",
        "Average precision": f"{metadata['average_precision']:.4f}",
        "Brier score": f"{metadata['brier_score']:.4f}",
        "Validation": metadata["validation_approach"],
        "Weights": metadata["weighting_approach"],
    }
    st.dataframe(pd.DataFrame(main.items(), columns=["Measure", "Result"]), hide_index=True, use_container_width=True)
    st.write(
        "Hospital-grouped validation reduces the risk that the model is evaluated on encounters from the same "
        "hospitals used for fitting, but it remains internal validation within the 2019 NEDS dataset."
    )
    if PERFORMANCE_PATH.is_file():
        sensitivity = pd.read_csv(PERFORMANCE_PATH)
        st.subheader("Primary and sensitivity analyses")
        st.dataframe(sensitivity, hide_index=True, use_container_width=True)
    else:
        st.error("The aggregate sensitivity-analysis report is missing.")
    st.warning("Calibration plots, confidence intervals, and complete subgroup performance analyses are not yet available.")

with about_tab:
    st.header("About the model")
    st.markdown(
        """
        - **Data source:** 2019 Nationwide Emergency Department Sample (NEDS).
        - **Cohort:** UC-related ED encounters with ICD-10-CM K51 in any diagnosis position.
        - **App endpoint:** admission to the same hospital (`EDevent == 2`). Undocumented outcomes were excluded.
        - **Secondary research endpoint:** admission, transfer to another short-term hospital, or ED death (`EDevent` in `{2, 3, 9}`). This app does **not** load that secondary model.
        - **Predictors:** demographics, administrative social/access fields, coded acute diagnoses, Elixhauser Refined comorbidities, and hospital characteristics.
        - **Area-level income input:** residential ZIP is looked up locally against 2019 ACS 5-year ZCTA median household income and converted with the official 2019 HCUP thresholds. The ZIP itself is not a model input or retained.
        - **Validation:** five-fold hospital-grouped internal validation with `DISCWT` weighting.
        """
    )
    with st.expander("Leakage variables excluded"):
        st.write(", ".join(metadata["leakage_exclusions"]))
    st.subheader("Limitations")
    for limitation in metadata["limitations"]:
        st.markdown(f"- {limitation}")
    st.markdown("- `ZIPINC_QRTL` is an area-level socioeconomic proxy, not individual household income; HCUP thresholds vary by year.")
    st.markdown("- The local ACS ZCTA lookup is an approximation of the original NEDS variable because NEDS used proprietary Claritas ZIP estimates, not ACS ZCTA estimates.")
    st.error("This research prototype must not be used to recommend admission, discharge, biologic therapy, or any other treatment.")
