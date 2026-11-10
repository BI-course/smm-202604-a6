"""
Streamlit Community Cloud app -- one tab per model, serving OUR GROUP'S four
models from the SAME trained .joblib artifacts that api.py serves:

    1. Regression       -- Nairobi monthly rental price      (Lasso, log1p target)
    2. Classification   -- Employee attrition (Yes/No)       (SMOTE + linear SVC)

Nothing is retrained here. The artifacts are read from the repository's
./model folder (the same folder api.py uses).

Run locally from the REPOSITORY ROOT (Streamlit Cloud also runs from there):
    streamlit run streamlit-sharing-using-streamlit/app.py
"""

import os

import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="BBT 4206 Group Models", page_icon="📊", layout="centered")

# Streamlit Cloud runs from the repository root, but resolving the path from
# this file means the app also works when started from any other folder.
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "model")



# Streamlit re-runs this whole script on every click; @st.cache_resource
# loads the four artifacts from disk only ONCE per server process.
@st.cache_resource
def load_artifacts():
    def load(filename):
        return joblib.load(os.path.join(MODEL_DIR, filename))

    return (
        load("lasso_regressor_for_rental_price.joblib"),
        load("svc_classifier_for_attrition.joblib"),
    )


try:
    rental_pipeline, attrition_pipeline = load_artifacts()
except FileNotFoundError as exc:
    st.error(f"Model file missing: {exc}. The ./model folder must be at the repository root.")
    st.stop()


def fitted_categories(pipeline, transformer_name, encoder_step):
    """Category values each pipeline was trained on -- used for the dropdowns."""
    branch = pipeline.named_steps["preprocessor"].named_transformers_[transformer_name]
    encoder = branch.named_steps[encoder_step]
    return {col: [str(v) for v in cats] for col, cats in zip(branch.feature_names_in_, encoder.categories_)}


RENTAL_FIELDS = list(rental_pipeline.feature_names_in_)
RENTAL_CATS = fitted_categories(rental_pipeline, "nominal", "onehot")
ATTRITION_FIELDS = list(attrition_pipeline.feature_names_in_)
ATTRITION_CATS = fitted_categories(attrition_pipeline, "cat", "encoder")


def select(label, options, default):
    return st.selectbox(label, options, index=options.index(default))


st.title("BBT 4206: Our Group's Models")
st.caption("One tab per model. Every tab uses the same saved `.joblib` artifacts as our Flask API; nothing is retrained.")

tab_reg, tab_clf = st.tabs([
    "Rental Price", "Employee Attrition",
])

# -----------------------------------------------------------------------
# 1. Regression -- monthly rental price
# -----------------------------------------------------------------------
with tab_reg:
    st.subheader("Monthly rental price (Regression)")
    with st.form("rental_form"):
        c1, c2 = st.columns(2)
        with c1:
            property_type = select("Property type", RENTAL_CATS["property_type"], "Apartment")
            furnishing_status = select("Furnishing", RENTAL_CATS["furnishing_status"], "Semi-Furnished")
            size_sqm = st.number_input("Size (sqm)", min_value=1.0, value=120.0)
            distance_to_cbd_km = st.number_input("Distance to CBD (km)", min_value=0.0, value=8.0)
            security_rating = st.number_input("Security rating (1-5)", min_value=1.0, max_value=5.0, value=4.0, step=0.5)
            nearby_schools_count = st.number_input("Nearby schools", min_value=0, value=3, step=1)
        with c2:
            neighborhood = select("Neighborhood", RENTAL_CATS["neighborhood"], "Kilimani")
            num_bedrooms = st.number_input("Bedrooms", min_value=0, value=3, step=1)
            property_age_years = st.number_input("Property age (years)", min_value=0.0, value=5.0)
            service_charge_kes = st.number_input("Service charge (KES)", min_value=0.0, value=15000.0, step=500.0)
            amenities_score = st.number_input("Amenities score (0-100)", min_value=0.0, max_value=100.0, value=65.0)
        submitted = st.form_submit_button("Predict monthly rent", type="primary")

    if submitted:
        row = pd.DataFrame([{
            "property_type": property_type, "neighborhood": neighborhood,
            "furnishing_status": furnishing_status, "property_age_years": property_age_years,
            "size_sqm": size_sqm, "num_bedrooms": num_bedrooms,
            "distance_to_cbd_km": distance_to_cbd_km, "service_charge_kes": service_charge_kes,
            "security_rating": security_rating, "amenities_score": amenities_score,
            "nearby_schools_count": nearby_schools_count,
        }])[RENTAL_FIELDS]
        rent_kes = float(np.expm1(rental_pipeline.predict(row)[0]))  # model predicts log1p(rent)
        st.metric("Predicted monthly rent", f"KES {rent_kes:,.0f}")
        st.caption("Model: Lasso Regressor (alpha=0.001, log1p target)")

# -----------------------------------------------------------------------
# 2. Classification -- employee attrition
# -----------------------------------------------------------------------
with tab_clf:
    st.subheader("Employee attrition (Classification)")
    with st.form("attrition_form"):
        c1, c2 = st.columns(2)
        with c1:
            department = select("Department", ATTRITION_CATS["department"], "Engineering")
            education_level = select("Education level", ATTRITION_CATS["education_level"], "Master's")
            years_at_company = st.number_input("Years at company", min_value=0.0, value=6.0)
            monthly_salary_kes = st.number_input("Monthly salary (KES)", min_value=0.0, value=95000.0, step=1000.0)
            job_satisfaction_score = st.number_input("Job satisfaction (1-5)", min_value=1.0, max_value=5.0, value=2.0)
            distance_from_home_km = st.number_input("Distance from home (km)", min_value=0.0, value=32.0)
            remote_work_days_per_week = st.number_input("Remote days / week", min_value=0, max_value=5, value=1, step=1)
        with c2:
            job_role_level = select("Job role level", ATTRITION_CATS["job_role_level"], "Mid")
            performance_rating = select("Performance rating", ATTRITION_CATS["performance_rating"], "High")
            years_since_last_promotion = st.number_input("Years since last promotion", min_value=0.0, value=3.5)
            overtime_hours_per_month = st.number_input("Overtime hours / month", min_value=0.0, value=45.0)
            work_life_balance_score = st.number_input("Work-life balance (1-5)", min_value=1.0, max_value=5.0, value=2.1)
            num_companies_worked_previously = st.number_input("Previous companies", min_value=0, value=3, step=1)
            bonus_amount_kes = st.number_input("Bonus (KES)", min_value=0.0, value=30000.0, step=1000.0)
        submitted = st.form_submit_button("Predict attrition", type="primary")

    if submitted:
        row = pd.DataFrame([{
            "department": department, "job_role_level": job_role_level,
            "education_level": education_level, "performance_rating": performance_rating,
            "years_at_company": years_at_company, "years_since_last_promotion": years_since_last_promotion,
            "monthly_salary_kes": monthly_salary_kes, "overtime_hours_per_month": overtime_hours_per_month,
            "job_satisfaction_score": job_satisfaction_score, "work_life_balance_score": work_life_balance_score,
            "distance_from_home_km": distance_from_home_km,
            "num_companies_worked_previously": num_companies_worked_previously,
            "remote_work_days_per_week": remote_work_days_per_week, "bonus_amount_kes": bonus_amount_kes,
        }])[ATTRITION_FIELDS]
        predicted = str(attrition_pipeline.predict(row)[0])
        probabilities = dict(zip(attrition_pipeline.named_steps["model"].classes_, attrition_pipeline.predict_proba(row)[0]))
        (st.error if predicted == "Yes" else st.success)(
            f"**{'Likely to leave' if predicted == 'Yes' else 'Likely to stay'}** "
            f"(probability of leaving: {probabilities['Yes']:.1%})"
        )
        st.caption("Model: Support Vector Classifier (linear, C=0.1, SMOTE-resampled at training only)")