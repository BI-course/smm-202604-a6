---
title: BBT 4206 Group Model Demos
emoji: 📊
colorFrom: indigo
colorTo: green
sdk: gradio
python_version: "3.12"
app_file: app.py
pinned: false
---

# BBT 4206: Our Group's Models (Gradio)

One Gradio app with one tab per model, serving the same trained artifacts as our Flask API (`api.py`):

| Tab | Task | Artifact |
|:----|:-----|:---------|
| Rental Price | Regression | `lasso_regressor_for_rental_price.joblib` |
| Employee Attrition | Classification | `svc_classifier_for_attrition.joblib` |
| Retail Store Segment | k-Means clustering | `kmeans_retail_store_segmentation.joblib` |
| Music Recommender | Apriori association rules | `music_streaming_arm_rules.joblib` |

Upload this `README.md`, `app.py`, `requirements.txt` and the four `.joblib` files to the root of the Space.
The YAML header above pins Python 3.12, which scikit-learn 1.9.0 (the version the models were saved with) requires.
