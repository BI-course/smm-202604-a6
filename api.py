"""
api.py

A RESTish API, built with Flask, that serves the machine learning artifacts
produced in earlier labs:

    1. Regression       -- SME monthly revenue prediction
    2. Classification   -- SME credit risk category
    3. Clustering       -- Mall customer segment assignment
    4. Association Rule -- Product recommendation from a cart
       Mining (ARM)
    5. Regression       -- Nairobi monthly rental price prediction
    6. Classification   -- Employee attrition (Yes/No)
    7. Clustering       -- Retail store segment assignment
    8. Association Rule -- Music track recommendation from a playlist
       Mining (ARM)        (+ the top mined rules, served from a CSV)

The point of this file is to show you that a fitted scikit-learn /
imbalanced-learn Pipeline is a SINGLE object that already knows how to impute
missing values, encode categorical columns, scale numeric columns, and (for
the classifier) apply SMOTE only during training -- never during inference.
Because all of that preprocessing lives inside the Pipeline itself, this file
never manually label-encodes anything, never manually engineers date features,
and never manually reorders columns before calling .predict(). It builds a
DataFrame straight from the incoming JSON and hands it to the pipeline as-is.
This is the main lesson of the lab: "the pipeline IS the deployment artifact."

Run this file directly to start a local development server:

    python api.py
"""

import os

import joblib
import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)

# -----------------------------------------------------------------------
# CORS (Cross-Origin Resource Sharing)
# -----------------------------------------------------------------------
# By default, a browser blocks JavaScript on one origin (e.g., a page served
# from http://127.0.0.1:5500) from calling an API on a different origin
# (e.g., http://127.0.0.1:5000). `flask_cors` relaxes that restriction for the
# specific origins we trust during local development.
#
# A few alternative configurations are shown below (commented out) so you can
# see how the restriction can be loosened or tightened depending on your
# deployment target. Only ONE of these should be active at a time.

# Option A (the most permissive. This is fine for a quick classroom demo or
# local development, *NEVER* for production):
# CORS(app)

# Option B (allow only a specific frontend origin):
# CORS(app, resources={r"/api/*": {"origins": "http://127.0.0.1:5500"}})

# Option C (the one actually in use here -- allow localhost/127.0.0.1 on the
# ports our lab's frontend files are typically served from):
CORS(app, resources={r"/api/*": {"origins": [
    "http://localhost:5000", "http://127.0.0.1:5000",
    "http://localhost:5500", "http://127.0.0.1:5500",
    "https://localhost:443", "https://127.0.0.1:443",
    "http://localhost:63342",  # VS Code Live Server extension
    "null",  # allows opening a frontend .html file directly (file://) with no web server
]}})

# -----------------------------------------------------------------------
# The best practice is to load the four persisted artifacts ONCE, at startup,
# not per-request.
# Loading inside a route function would re-read the file from disk on every
# single API call, which is slow and pointless -- the fitted model never
# changes while the server is running.
# -----------------------------------------------------------------------
# Resolved relative to this file (not the current working directory), so the
# API finds its models whether it is started from the repo root, from another
# folder, by Gunicorn inside Docker, or by Render.
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")

regression_pipeline = joblib.load(
    os.path.join(MODEL_DIR, "lasso_regressor_for_sme_revenue.joblib")
)

classification_pipeline = joblib.load(
    os.path.join(MODEL_DIR, "svc_classifier_for_sme_credit_risk.joblib")
)

clustering_artifact = joblib.load(
    os.path.join(MODEL_DIR, "kmeans_mall_customer_segmentation.joblib")
)
clustering_scaler = clustering_artifact["scaler"]
clustering_kmeans = clustering_artifact["kmeans"]

arm_artifact = joblib.load(
    os.path.join(MODEL_DIR, "apriori_recommendation_rules.joblib")
)
recommendation_lookup = arm_artifact["recommendation_lookup"]
sku_lookup = arm_artifact["sku_lookup"]

print("The four lab-demo artifacts have been loaded.")


# -----------------------------------------------------------------------
# OUR GROUP'S FOUR MODELS (rental price, attrition, retail stores, music)
# -----------------------------------------------------------------------
# All four were trained in our earlier labs and saved with scikit-learn
# 1.9.0 -- the exact version pinned in requirements/constraints.txt. A
# pickled scikit-learn object is only guaranteed to work with the SAME
# version that saved it, so local, Docker, Render, Hugging Face and
# Streamlit all install that same pinned version.
#
# They are loaded defensively: if one of them cannot be loaded, the API
# still starts and only that model's endpoints answer with HTTP 503 (and the
# reason), instead of the whole server failing to boot.
unavailable_models = {}


def load_optional_artifact(name, filename):
    try:
        return joblib.load(os.path.join(MODEL_DIR, filename))
    except Exception as exc:  # noqa: BLE001 -- report any load failure, keep serving the rest
        unavailable_models[name] = f"{filename} could not be loaded: {type(exc).__name__}: {exc}"
        print(f"WARNING: {unavailable_models[name]}")
        return None


def unavailable_error(name):
    return jsonify({"error": "Model is not available.", "detail": unavailable_models[name]}), 503


rental_pipeline = load_optional_artifact(
    "rental", "lasso_regressor_for_rental_price.joblib"
)

attrition_pipeline = load_optional_artifact(
    "attrition", "svc_classifier_for_attrition.joblib"
)

store_clustering_artifact = load_optional_artifact(
    "store", "kmeans_retail_store_segmentation.joblib"
)

music_arm_artifact = load_optional_artifact(
    "music", "music_streaming_arm_rules.joblib"
)

try:
    top_association_rules = pd.read_csv(os.path.join(MODEL_DIR, "top_association_rules.csv"))
except Exception as exc:  # noqa: BLE001
    top_association_rules = None
    unavailable_models["top_rules"] = f"top_association_rules.csv could not be loaded: {exc}"
    print(f"WARNING: {unavailable_models['top_rules']}")

print("The API is ready." + (f" Unavailable: {sorted(unavailable_models)}" if unavailable_models else ""))


# -----------------------------------------------------------------------
# Input metadata read FROM THE FITTED MODELS themselves
# -----------------------------------------------------------------------
# Rather than hard-coding which fields are required and which category
# values are valid, read them from what each pipeline learned at fit time.
# If a model is retrained with a new category, validation updates itself.
def fitted_categories(pipeline, transformer_name, encoder_step):
    """{column: [allowed values]} for one categorical branch of a fitted pipeline."""
    branch = pipeline.named_steps["preprocessor"].named_transformers_[transformer_name]
    encoder = branch.named_steps[encoder_step]
    columns = branch.feature_names_in_
    return {col: [str(v) for v in cats] for col, cats in zip(columns, encoder.categories_)}


RENTAL_FIELDS, RENTAL_CATEGORIES, RENTAL_NULLABLE = [], {}, ["service_charge_kes"]
if rental_pipeline is not None:
    RENTAL_FIELDS = list(rental_pipeline.feature_names_in_)
    RENTAL_CATEGORIES = fitted_categories(rental_pipeline, "nominal", "onehot")

ATTRITION_FIELDS, ATTRITION_CATEGORIES = [], {}
if attrition_pipeline is not None:
    ATTRITION_FIELDS = list(attrition_pipeline.feature_names_in_)
    ATTRITION_CATEGORIES = fitted_categories(attrition_pipeline, "cat", "encoder")

STORE_FIELDS = store_clustering_artifact["feature_names"] if store_clustering_artifact else []

# Business-facing names for each retail-store cluster, written from the
# cluster centres (original units) the K-Means model learned:
#   cluster | revenue (KES) | foot traffic | staff | size (sqm) | basket (KES)
#      0    |    477,524    |     179      |   4   |    110     |      92
#      1    |  4,680,043    |     897      |  22   |    854     |     176
#      2    |  1,109,766    |     507      |   9   |    285     |      95
#      3    |  1,097,548    |     148      |   5   |    108     |     261
STORE_SEGMENTS = {
    0: {"name": "Small Neighbourhood Stores",
        "description": "Low revenue, small floor space and few staff; customers buy small baskets."},
    1: {"name": "Flagship Superstores",
        "description": "Highest revenue, foot traffic, staff and floor space by a wide margin."},
    2: {"name": "High-Traffic Value Stores",
        "description": "Mid-size stores whose revenue comes from many shoppers buying small baskets."},
    3: {"name": "Premium Boutiques",
        "description": "Small stores with few shoppers but the largest baskets, earning mid-level revenue."},
}


# -----------------------------------------------------------------------
# Basic error handling for missing input
# -----------------------------------------------------------------------
# Returns (payload, None) if the JSON body is present and contains every
# required field, otherwise (None, error_response) with HTTP 400 so the
# client gets a clear message instead of a 500 Internal Server Error.
def get_payload_or_error(required_fields):
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict) or not payload:
        return None, (jsonify({"error": "Request body must be a non-empty JSON object."}), 400)
    missing = [field for field in required_fields if field not in payload]
    if missing:
        return None, (jsonify({"error": "Missing required field(s).", "missing_fields": missing}), 400)
    return payload, None


# Checks the VALUES (not just the presence) of the fields our group's models
# need, so a typo such as "Kilimanii" or a number sent as "abc" gets a clear
# HTTP 400 that says exactly what to fix, instead of a silent wrong answer
# or a 500 Internal Server Error from deep inside scikit-learn.
def validate_values(payload, numeric_fields=(), categories=None, nullable=()):
    invalid = {}
    for field in numeric_fields:
        value = payload.get(field)
        if value is None:
            if field not in nullable:
                invalid[field] = "must be a number (null is not allowed for this field)"
        elif isinstance(value, bool) or not isinstance(value, (int, float)):
            invalid[field] = f"must be a number, got {value!r}"
        elif value < 0:
            invalid[field] = f"must not be negative, got {value}"
    for field, allowed in (categories or {}).items():
        if payload.get(field) not in allowed:
            invalid[field] = f"must be one of {allowed}, got {payload.get(field)!r}"
    if invalid:
        return jsonify({"error": "Invalid field value(s).", "invalid_fields": invalid}), 400
    return None


# JSON (instead of Flask's default HTML pages) for errors raised outside a
# route -- e.g. a mistyped URL or a GET sent to a POST-only endpoint.
@app.errorhandler(404)
def not_found(_error):
    return jsonify({"error": "Endpoint not found.", "hint": "GET /api/v1/models lists every endpoint."}), 404


@app.errorhandler(405)
def method_not_allowed(_error):
    return jsonify({"error": "HTTP method not allowed for this endpoint."}), 405


@app.errorhandler(500)
def internal_error(_error):
    return jsonify({"error": "The server failed to process this request."}), 500


# 1. REGRESSION -- predict monthly revenue (KES) for an SME

# Sample request (cURL):
#   curl -X POST http://127.0.0.1:5000/api/v1/models/sme-revenue-regressor/predictions \
#        -H "Content-Type: application/json" \
#        -d '{"sector": "Technology", "county": "Nairobi", "quarter": "Q3",
#             "business_age_years": 4.5, "num_employees": 12,
#             "owner_education_level": "Secondary",
#             "monthly_marketing_spend_kes": 45000,
#             "avg_daily_foot_traffic": null,
#             "online_presence_score": 78.0, "loan_amount_kes": 250000,
#             "interest_rate_pct": 14.5, "inflation_rate_pct": 6.5,
#             "competitor_density": 5, "customer_satisfaction_score": 4.2}'
#
# Sample request (PowerShell):
#   Invoke-RestMethod -Method Post `
#     -Uri "http://127.0.0.1:5000/api/v1/models/sme-revenue-regressor/predictions" `
#     -ContentType "application/json" `
#     -Body (@{sector="Technology"; county="Nairobi"; quarter="Q3";
#              business_age_years=4.5; num_employees=12;
#              owner_education_level="Secondary";
#              monthly_marketing_spend_kes=45000; avg_daily_foot_traffic=$null;
#              online_presence_score=78.0; loan_amount_kes=250000;
#              interest_rate_pct=14.5; inflation_rate_pct=6.5;
#              competitor_density=5; customer_satisfaction_score=4.2} | ConvertTo-Json)
#
# Note: avg_daily_foot_traffic is deliberately null for a Technology-sector
# business (fully online -- foot traffic is structurally not applicable, not
# missing at random). The pipeline's own imputer handles the null value; nothing
# in this file needs to know that rule.
@app.route("/api/v1/models/sme-revenue-regressor/predictions", methods=["POST"])
def predict_revenue():
    payload, error = get_payload_or_error(["sector", "county", "business_age_years", "num_employees"])
    if error:
        return error

    # Build a one-row DataFrame straight from the JSON body. No manual
    # label-encoding, no manual date parsing, no manual column reordering, etc. --
    # the pipeline's ColumnTransformer already knows which columns it expects
    # and how to preprocess each one.
    input_row = pd.DataFrame([payload])

    predicted_log_revenue = regression_pipeline.predict(input_row)[0]
    # The pipeline was trained on log1p(revenue) because revenue was
    # right-skewed; we invert that single transformation here to report the
    # predictions in KES.
    import numpy as np
    predicted_revenue_kes = float(np.expm1(predicted_log_revenue))

    return jsonify({
        "predicted_monthly_revenue_kes": round(predicted_revenue_kes, 2),
        "model": "Lasso Regressor",
    })


# 2. CLASSIFICATION -- predict SME credit risk category (Low/Medium/High)

# Sample request (cURL):
#   curl -X POST http://127.0.0.1:5000/api/v1/models/sme-credit-risk-classifier/predictions \
#        -H "Content-Type: application/json" \
#        -d '{"sector": "Retail", "county": "Nairobi", "quarter": "Q2",
#             "loan_purpose": "Working Capital", "owner_education_level": "Secondary",
#             "business_age_years": 2.1, "num_employees": 4,
#             "monthly_revenue_kes": 850000, "loan_amount_kes": 150000,
#             "loan_term_months": 12, "interest_rate_pct": 16.5,
#             "collateral_value_kes": null, "debt_to_income_ratio": 0.62,
#             "credit_score": 590, "num_previous_loans": 1,
#             "missed_payments_count": 3, "online_presence_score": 40.0,
#             "customer_satisfaction_score": 3.4, "competitor_density": 6}'
#
#
# Note: collateral_value_kes is null here because "Working Capital" loans are
# more often unsecured (structural missingness, not missing at random) --
# again, the pipeline's imputer + missingness indicator already handles this
# internally; this route does not need special code for it.
@app.route("/api/v1/models/sme-credit-risk-classifier/predictions", methods=["POST"])
def predict_credit_risk():
    payload, error = get_payload_or_error(["sector", "county", "loan_amount_kes", "credit_score"])
    if error:
        return error
    input_row = pd.DataFrame([payload])

    predicted_class = classification_pipeline.predict(input_row)[0]
    predicted_probabilities = classification_pipeline.predict_proba(input_row)[0]
    class_labels = classification_pipeline.named_steps["model"].classes_

    return jsonify({
        "predicted_credit_risk_category": str(predicted_class),
        "class_probabilities": {
            str(label): round(float(prob), 4)
            for label, prob in zip(class_labels, predicted_probabilities)
        },
        "model": "Support Vector Classifier (tuned, SMOTE-resampled at training only)",
    })


# 3. CLUSTERING -- assign a new mall customer to an existing segment

# Sample request (cURL):
#   curl -X POST http://127.0.0.1:5000/api/v1/models/mall-customer-segmenter/predictions \
#        -H "Content-Type: application/json" \
#        -d '{"Age": 27, "Annual Income (k$)": 45, "Spending Score (1-100)": 72}'
#
# Note: unlike the two Pipelines above, K-Means does not bundle its own
# scaler, so this is the one endpoint where the scaling step is applied
# explicitly (scaler.transform(...) then kmeans.predict(...)) -- exactly the
# two-object pattern the clustering notebook persisted together for this
# reason. There is still no manual encoding or feature engineering: the
# three numeric fields are used exactly as the client sends them.
@app.route("/api/v1/models/mall-customer-segmenter/predictions", methods=["POST"])
def predict_customer_segment():
    payload, error = get_payload_or_error(["Age", "Annual Income (k$)", "Spending Score (1-100)"])
    if error:
        return error
    input_row = pd.DataFrame([payload])

    scaled_input = clustering_scaler.transform(input_row)
    assigned_cluster = int(clustering_kmeans.predict(scaled_input)[0])

    return jsonify({
        "assigned_cluster": assigned_cluster,
        "model": "K-Means (k=5, standardized features)",
    })


# 4. ASSOCIATION RULE MINING -- recommend products for a shopping cart

# Sample request (cURL):
#   curl -X POST http://127.0.0.1:5000/api/v1/models/ecommerce-recommender/predictions \
#        -H "Content-Type: application/json" \
#        -d '{"cart": ["H06", "H07"]}'
#
#   curl -X POST http://127.0.0.1:5000/api/v1/models/ecommerce-recommender/predictions \
#        -H "Content-Type: application/json" \
#        -d '{"cart": ["P04", "P01"]}'
#
# Note: there is no scikit-learn ".predict()" here -- the mined association
# rules were precomputed into a lookup dictionary
# (antecedent SKU-set -> ranked candidate consequents)
# during the notebook's persistence stage. This
# endpoint performs the exact same lookup+rank step a production recommender
# microservice would perform on every incoming request; it does not mine
# for the rules again.
@app.route("/api/v1/models/ecommerce-recommender/predictions", methods=["POST"])
def recommend_products():
    payload, error = get_payload_or_error(["cart"])
    if error:
        return error
    if not isinstance(payload["cart"], list) or not payload["cart"]:
        return jsonify({"error": "'cart' must be a non-empty list of SKUs."}), 400
    cart = set(payload["cart"])
    top_n = int(payload.get("top_n", 3))

    candidates = {}
    for antecedent, consequent_list in recommendation_lookup.items():
        if antecedent.issubset(cart):
            for sku, confidence, lift in consequent_list:
                if sku not in cart:
                    if sku not in candidates or confidence > candidates[sku][0]:
                        candidates[sku] = (confidence, lift)

    ranked = sorted(candidates.items(), key=lambda item: item[1][0], reverse=True)[:top_n]
    recommendations = [
        {
            "sku": str(sku),
            "product_name": sku_lookup.loc[sku, "product_name"],
            "confidence": round(float(confidence), 3),
            "lift": round(float(lift), 2),
        }
        for sku, (confidence, lift) in ranked
    ]

    return jsonify({
        "cart": sorted(cart),
        "recommendations": recommendations,
        "model": "Apriori association rules, cleaned + precomputed lookup",
    })


# =======================================================================
# OUR GROUP'S FOUR MODELS
# =======================================================================

# 5. REGRESSION -- predict monthly rental price (KES) for a Nairobi property

# Sample request (cURL):
#   curl -X POST http://127.0.0.1:5000/api/v1/models/rental-price-regressor/predictions \
#        -H "Content-Type: application/json" \
#        -d '{"property_type": "Apartment", "neighborhood": "Kilimani",
#             "furnishing_status": "Semi-Furnished", "property_age_years": 5.0,
#             "size_sqm": 120.0, "num_bedrooms": 3, "distance_to_cbd_km": 8.0,
#             "service_charge_kes": 15000.0, "security_rating": 4.0,
#             "amenities_score": 65.0, "nearby_schools_count": 3}'
#
# Sample request (PowerShell):
#   Invoke-RestMethod -Method Post `
#     -Uri "http://127.0.0.1:5000/api/v1/models/rental-price-regressor/predictions" `
#     -ContentType "application/json" `
#     -Body (@{property_type="Apartment"; neighborhood="Kilimani";
#              furnishing_status="Semi-Furnished"; property_age_years=5.0;
#              size_sqm=120.0; num_bedrooms=3; distance_to_cbd_km=8.0;
#              service_charge_kes=15000.0; security_rating=4.0;
#              amenities_score=65.0; nearby_schools_count=3} | ConvertTo-Json)
#
# Note: service_charge_kes and security_rating may be sent as null -- the
# pipeline's own median imputers fill them, exactly as during training.
RENTAL_NULLABLE.append("security_rating")


@app.route("/api/v1/models/rental-price-regressor/predictions", methods=["POST"])
def predict_rental_price():
    if rental_pipeline is None:
        return unavailable_error("rental")
    payload, error = get_payload_or_error(RENTAL_FIELDS)
    if error:
        return error
    numeric_fields = [f for f in RENTAL_FIELDS if f not in RENTAL_CATEGORIES]
    error = validate_values(payload, numeric_fields, RENTAL_CATEGORIES, nullable=RENTAL_NULLABLE)
    if error:
        return error

    input_row = pd.DataFrame([payload])[RENTAL_FIELDS]

    # The pipeline was trained on log1p(monthly rent) because rent is
    # right-skewed; expm1 converts the prediction back to KES.
    import numpy as np
    predicted_log_rent = rental_pipeline.predict(input_row)[0]
    predicted_rent_kes = float(np.expm1(predicted_log_rent))

    return jsonify({
        "predicted_monthly_rent_kes": round(predicted_rent_kes, 2),
        "model": "Lasso Regressor (alpha=0.001, log1p target)",
    })


# 6. CLASSIFICATION -- predict whether an employee will leave (Yes/No)

# Sample request (cURL):
#   curl -X POST http://127.0.0.1:5000/api/v1/models/employee-attrition-classifier/predictions \
#        -H "Content-Type: application/json" \
#        -d '{"department": "Engineering", "job_role_level": "Mid",
#             "education_level": "Master'"'"'s", "performance_rating": "High",
#             "years_at_company": 6.0, "years_since_last_promotion": 3.5,
#             "monthly_salary_kes": 95000, "overtime_hours_per_month": 45.0,
#             "job_satisfaction_score": 2.0, "work_life_balance_score": 2.1,
#             "distance_from_home_km": 32.0, "num_companies_worked_previously": 3,
#             "remote_work_days_per_week": 1, "bonus_amount_kes": 30000}'
#
# Note: as with the credit-risk classifier, SMOTE lives inside the
# imbalanced-learn Pipeline and is skipped automatically at predict time.
# bonus_amount_kes and work_life_balance_score may be null (they had gaps in
# the training data, which the pipeline's median imputer fills).
ATTRITION_NULLABLE = ["bonus_amount_kes", "work_life_balance_score"]


@app.route("/api/v1/models/employee-attrition-classifier/predictions", methods=["POST"])
def predict_attrition():
    if attrition_pipeline is None:
        return unavailable_error("attrition")
    payload, error = get_payload_or_error(ATTRITION_FIELDS)
    if error:
        return error
    numeric_fields = [f for f in ATTRITION_FIELDS if f not in ATTRITION_CATEGORIES]
    error = validate_values(payload, numeric_fields, ATTRITION_CATEGORIES, nullable=ATTRITION_NULLABLE)
    if error:
        return error

    input_row = pd.DataFrame([payload])[ATTRITION_FIELDS]

    predicted_class = attrition_pipeline.predict(input_row)[0]
    predicted_probabilities = attrition_pipeline.predict_proba(input_row)[0]
    class_labels = attrition_pipeline.named_steps["model"].classes_

    return jsonify({
        "predicted_attrition": str(predicted_class),
        "class_probabilities": {
            str(label): round(float(prob), 4)
            for label, prob in zip(class_labels, predicted_probabilities)
        },
        "model": "Support Vector Classifier (linear, C=0.1, SMOTE-resampled at training only)",
    })


# 7. CLUSTERING -- assign a new retail store to an existing segment

# Sample request (cURL):
#   curl -X POST http://127.0.0.1:5000/api/v1/models/retail-store-segmenter/predictions \
#        -H "Content-Type: application/json" \
#        -d '{"monthly_revenue_kes": 850000, "daily_foot_traffic": 320,
#             "staff_count": 6, "store_size_sqm": 220, "avg_basket_size_kes": 105}'
#
# Note: like the mall segmenter, K-Means does not bundle its scaler, so the
# saved scaler is applied explicitly first. The artifact also stores its
# feature_names, so the columns are put in the exact order the scaler was
# fitted on, whatever order the client sends them in.
@app.route("/api/v1/models/retail-store-segmenter/predictions", methods=["POST"])
def predict_store_segment():
    if store_clustering_artifact is None:
        return unavailable_error("store")
    payload, error = get_payload_or_error(STORE_FIELDS)
    if error:
        return error
    error = validate_values(payload, STORE_FIELDS)
    if error:
        return error

    input_row = pd.DataFrame([payload])[STORE_FIELDS]
    scaled_input = store_clustering_artifact["scaler"].transform(input_row)
    kmeans = store_clustering_artifact["kmeans"]
    assigned_cluster = int(kmeans.predict(scaled_input)[0])
    segment = STORE_SEGMENTS.get(assigned_cluster, {})

    return jsonify({
        "assigned_cluster": assigned_cluster,
        "segment_name": segment.get("name"),
        "segment_description": segment.get("description"),
        "model": f"K-Means (k={kmeans.n_clusters}, standardized features)",
    })


# 8. ASSOCIATION RULE MINING -- recommend music tracks for a listening session

# Sample request (cURL):
#   curl -X POST http://127.0.0.1:5000/api/v1/models/music-recommender/predictions \
#        -H "Content-Type: application/json" \
#        -d '{"playlist": ["M09", "M10"]}'
#
#   curl -X POST http://127.0.0.1:5000/api/v1/models/music-recommender/predictions \
#        -H "Content-Type: application/json" \
#        -d '{"playlist": ["M11"], "top_n": 2}'
#
# Same lookup+rank approach as the e-commerce recommender above: the rules
# were mined once in the notebook and precomputed into a lookup; nothing is
# mined here.
@app.route("/api/v1/models/music-recommender/predictions", methods=["POST"])
def recommend_tracks():
    if music_arm_artifact is None:
        return unavailable_error("music")
    payload, error = get_payload_or_error(["playlist"])
    if error:
        return error
    if not isinstance(payload["playlist"], list) or not payload["playlist"]:
        return jsonify({"error": "'playlist' must be a non-empty list of track IDs, e.g. [\"M09\", \"M10\"]."}), 400

    track_lookup = music_arm_artifact["track_lookup"]
    playlist = {str(track).strip().upper() for track in payload["playlist"]}
    unknown = sorted(playlist - set(track_lookup.index))
    if unknown:
        return jsonify({
            "error": "Unknown track ID(s).",
            "unknown_tracks": unknown,
            "hint": "GET /api/v1/models/music-recommender/tracks lists every valid track ID.",
        }), 400
    try:
        top_n = int(payload.get("top_n", 3))
    except (TypeError, ValueError):
        return jsonify({"error": "'top_n' must be a whole number."}), 400
    if not 1 <= top_n <= 20:
        return jsonify({"error": "'top_n' must be between 1 and 20."}), 400

    candidates = {}
    for antecedent, consequent_list in music_arm_artifact["recommendation_lookup"].items():
        if antecedent.issubset(playlist):
            for track, confidence, lift in consequent_list:
                if track not in playlist:
                    if track not in candidates or confidence > candidates[track][0]:
                        candidates[track] = (confidence, lift)

    ranked = sorted(candidates.items(), key=lambda item: item[1][0], reverse=True)[:top_n]
    recommendations = [
        {
            "track_id": str(track),
            "item_name": str(track_lookup.loc[track, "item_name"]),
            "category": str(track_lookup.loc[track, "category"]),
            "confidence": round(float(confidence), 3),
            "lift": round(float(lift), 2),
        }
        for track, (confidence, lift) in ranked
    ]

    response = {
        "playlist": sorted(playlist),
        "recommendations": recommendations,
        "model": "Apriori association rules (music streaming), precomputed lookup",
    }
    if not recommendations:
        response["note"] = "No mined rule starts from this playlist. Try e.g. [\"M09\", \"M10\"] or [\"M11\"]."
    return jsonify(response)


# 8b. The music track catalogue (the valid IDs for the recommender above)

# Sample request (cURL):
#   curl http://127.0.0.1:5000/api/v1/models/music-recommender/tracks
@app.route("/api/v1/models/music-recommender/tracks", methods=["GET"])
def list_tracks():
    if music_arm_artifact is None:
        return unavailable_error("music")
    tracks = music_arm_artifact["track_lookup"].sort_index()
    return jsonify({
        "count": len(tracks),
        "tracks": [
            {"track_id": str(track_id), "item_name": str(row.item_name),
             "category": str(row.category), "duration_minutes": int(row.duration_minutes)}
            for track_id, row in tracks.iterrows()
        ],
    })


# 8c. The top mined association rules (read-only, from top_association_rules.csv)

# Sample request (cURL):
#   curl "http://127.0.0.1:5000/api/v1/models/music-recommender/rules?limit=5"
@app.route("/api/v1/models/music-recommender/rules", methods=["GET"])
def list_top_rules():
    if top_association_rules is None:
        return unavailable_error("top_rules")
    limit = request.args.get("limit", default=len(top_association_rules), type=int)
    columns = ["antecedents", "consequents", "support", "confidence", "lift"]
    rows = top_association_rules[columns].head(max(limit, 0))
    return jsonify({
        "count": len(rows),
        "rules": [
            {
                "antecedents": [s.strip() for s in str(r.antecedents).split(",")],
                "consequents": [s.strip() for s in str(r.consequents).split(",")],
                "support": round(float(r.support), 4),
                "confidence": round(float(r.confidence), 4),
                "lift": round(float(r.lift), 2),
            }
            for r in rows.itertuples(index=False)
        ],
    })


# -----------------------------------------------------------------------
# API discovery and health -- "the API is the product"
# -----------------------------------------------------------------------
# GET /api/v1/models documents every endpoint, the fields it needs and the
# exact category values our models accept, so a developer can integrate
# without reading this file. GET /api/v1/health reports whether each model
# loaded (useful as Render's health check path).
@app.route("/api/v1/models", methods=["GET"])
def list_models():
    return jsonify({
        "group_models": [
            {"task": "regression",
             "endpoint": "POST /api/v1/models/rental-price-regressor/predictions",
             "required_fields": RENTAL_FIELDS, "nullable_fields": RENTAL_NULLABLE,
             "allowed_values": RENTAL_CATEGORIES},
            {"task": "classification",
             "endpoint": "POST /api/v1/models/employee-attrition-classifier/predictions",
             "required_fields": ATTRITION_FIELDS, "nullable_fields": ATTRITION_NULLABLE,
             "allowed_values": ATTRITION_CATEGORIES},
            {"task": "clustering",
             "endpoint": "POST /api/v1/models/retail-store-segmenter/predictions",
             "required_fields": STORE_FIELDS,
             "segments": {str(k): v["name"] for k, v in STORE_SEGMENTS.items()}},
            {"task": "association rules",
             "endpoint": "POST /api/v1/models/music-recommender/predictions",
             "required_fields": ["playlist"], "optional_fields": {"top_n": "1-20, default 3"},
             "related": ["GET /api/v1/models/music-recommender/tracks",
                         "GET /api/v1/models/music-recommender/rules?limit=N"]},
        ],
        "lab_demo_models": [
            "POST /api/v1/models/sme-revenue-regressor/predictions",
            "POST /api/v1/models/sme-credit-risk-classifier/predictions",
            "POST /api/v1/models/mall-customer-segmenter/predictions",
            "POST /api/v1/models/ecommerce-recommender/predictions",
        ],
    })


@app.route("/api/v1/health", methods=["GET"])
def health():
    healthy = not unavailable_models
    return jsonify({
        "status": "ok" if healthy else "degraded",
        "unavailable": unavailable_models,
    }), 200 if healthy else 503


# -----------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------
if __name__ == "__main__":
    # Development server. debug=True gives auto-reload and an interactive
    # debugger on errors -- convenient in class, but NEVER use debug=True in
    # a deployed/public environment (it can expose a remote code execution
    # console to anyone who can reach the server).
    app.run(debug=True)

    # Production-style alternative (no debugger, no auto-reload):
    # app.run(debug=False, host="0.0.0.0", port=5000)

    # HTTPS/TLS alternative for local development (requires a self-signed
    # certificate; see app_server_reverse_proxy_server_setup.md):
    # app.run(debug=False, ssl_context=("cert.pem", "key.pem"))
