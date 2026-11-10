"""
Streamlit Community Cloud app -- one tab per model, serving OUR GROUP'S four
models from the SAME trained .joblib artifacts that api.py serves:

    1. Regression       -- Nairobi monthly rental price      (Lasso, log1p target)

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
    )


try:
    (rental_pipeline,) = load_artifacts()
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


def select(label, options, default):
    return st.selectbox(label, options, index=options.index(default))


st.title("BBT 4206: Our Group's Models")
st.caption("One tab per model. Every tab uses the same saved `.joblib` artifacts as our Flask API; nothing is retrained.")

(tab_reg,) = st.tabs(["Rental Price"])

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

