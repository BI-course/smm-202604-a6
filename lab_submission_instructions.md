# Lab Submission Instructions

## Student Details and Individual Member Contributions

**Name of the team on GitHub Classroom:** Vin Wonder, Collins Mwenda, Celestine Nyingi, Aguer Deng, Turkmen Desmond

**GitHub accounts:** Vin Wonder ([@VinWonder](https://github.com/VinWonder)), Collins Mwenda ([@collo7c](https://github.com/collo7c)), Celestine Nyingi([@nyingicele](https://github.com/nyingicele)), Aguer Deng ([@Aguer20](https://github.com/Aguer20)), Turkmen Desmond ([@D3zzm4nd](https://github.com/D3zzm4nd))

**How the work was divided:** each of Members 1-4 owns one of our four models from end to end: its API endpoint, its HTML page, its tab in the Gradio app and its tab in the Streamlit app, plus one piece of the shared API error handling. Member 5 owns deployment: Docker + Gunicorn + Nginx, Render, and publishing the Hugging Face Space and Streamlit app that Members 1-4 built tabs for. Each member worked on one feature branch, opened a pull request into `main`, and had it reviewed and approved by another member before merging. The branches were merged in the order Member 1 to Member 5, because each builds on the shared `api.py` and app files from the one before.

**Member 1:**

| **Details**                                                                                                                           | **Comment** |
|:--------------------------------------------------------------------------------------------------------------------------------------|:------------|
| **Student ID**                                                                                                                        | 169912 |
| **Name**                                                                                                                              | Vin Wonder |
| **What part of the lab did you personally<br/>contribute to (provide a link to the<br/>branch(es)), and what did you learn from it?** | **Part:** Part A, Regression (rental price), end to end. Added `POST /api/v1/models/rental-price-regressor/predictions`, which loads `model/lasso_regressor_for_rental_price.joblib` (Lasso, log1p target) and converts the prediction back to KES with `expm1`. Built its HTML page `frontend/rental_price_predictor.html` and its "Rental Price" tab in both the Gradio and Streamlit apps. Shared work: the basic error handling for missing input (`get_payload_or_error`, used by every endpoint, so a missing field or empty body returns HTTP 400 instead of a 500), the value-validation helper (`validate_values`) that returns HTTP 400 for wrong types, negative numbers and unknown categories, and the frontend landing page `frontend/index.html`.<br/>**Branch link:** https://github.com/BI-course/smm-202604-a6/tree/feature/regression-rental-price<br/>**What I learned:** The fitted Pipeline is the deployment artifact. Imputing, scaling and one-hot encoding all happen inside it, so the API only builds a DataFrame and calls `.predict()`. The model was trained on `log1p(rent)`, so the API must apply `expm1` or it returns nonsense values. Reading the allowed categories from the fitted encoder means a misspelt neighbourhood gets a clear 400 instead of a silently wrong price. |

**Member 2:**

| **Details**                                                                                                                           | **Comment** |
|:--------------------------------------------------------------------------------------------------------------------------------------|:------------|
| **Student ID**                                                                                                                        | 169424 |
| **Name**                                                                                                                              | Collins Mwenda |
| **What part of the lab did you personally<br/>contribute to (provide a link to the<br/>branch(es)), and what did you learn from it?** | **Part:** Part A, Classification (employee attrition), end to end. Added `POST /api/v1/models/employee-attrition-classifier/predictions`, which loads `model/svc_classifier_for_attrition.joblib` (SMOTE + linear SVC) and returns the Yes/No prediction with class probabilities. Built its HTML page `frontend/employee_attrition_classifier.html` and its "Employee Attrition" tab in both the Gradio and Streamlit apps. Shared work: JSON error responses for 404 (unknown URL), 405 (wrong HTTP method) and 500, instead of Flask's default HTML pages.<br/>**Branch link:** https://github.com/BI-course/smm-202604-a6/tree/feature/classification-attrition<br/>**What I learned:** SMOTE sits inside the imbalanced-learn Pipeline but only runs during `.fit()`, so the served model never resamples a client's request. Returning class probabilities, not just the label, shows how confident the model is. An API should answer every error in JSON, because a client program cannot read an HTML error page. |

**Member 3:**

| **Details**                                                                                                                           | **Comment** |
|:--------------------------------------------------------------------------------------------------------------------------------------|:------------|
| **Student ID**                                                                                                                        | 134779 |
| **Name**                                                                                                                              | Celestine Njiru |
| **What part of the lab did you personally<br/>contribute to (provide a link to the<br/>branch(es)), and what did you learn from it?** | **Part:** Part B, Clustering (retail store segments), end to end. Added `POST /api/v1/models/retail-store-segmenter/predictions`, which loads `model/kmeans_retail_store_segmentation.joblib` (saved scaler + K-Means, k=4), applies the scaler before predicting, and returns the cluster with a business-friendly segment name. Built its HTML page `frontend/retail_store_segmenter.html` and its "Retail Store Segment" tab in both the Gradio and Streamlit apps. Shared work: `GET /api/v1/health`, which reports whether every model loaded, and the start-up handling that keeps the API running (with a 503 for the affected endpoint) if one model file fails to load.<br/>**Branch link:** https://github.com/BI-course/smm-202604-a6/tree/feature/clustering-retail-stores<br/>**What I learned:** K-Means does not bundle its own scaler, so the scaler fitted during training has to be saved alongside the model and applied first. Otherwise revenue in KES swamps every other feature and every store lands in the wrong cluster. Saving the feature order with the model lets the API reorder the incoming columns safely. A bare cluster number means little to a business user, so the API also returns a segment name based on the cluster centres. |

**Member 4:**

| **Details**                                                                                                                           | **Comment** |
|:--------------------------------------------------------------------------------------------------------------------------------------|:------------|
| **Student ID**                                                                                                                        | 167745 |
| **Name**                                                                                                                              | Aguer Deng |
| **What part of the lab did you personally<br/>contribute to (provide a link to the<br/>branch(es)), and what did you learn from it?** | **Part:** Part B, Association Rule Mining (music recommendations), end to end. Added `POST /api/v1/models/music-recommender/predictions`, which loads `model/music_streaming_arm_rules.joblib` (precomputed Apriori lookup) and ranks recommended tracks by confidence, plus `GET /api/v1/models/music-recommender/tracks` and `GET /api/v1/models/music-recommender/rules` (served from `model/top_association_rules.csv`). Built its HTML page `frontend/music_recommender.html` and its "Music Recommender" tab in both the Gradio and Streamlit apps. Shared work: `GET /api/v1/models`, which documents every endpoint, its required fields and the allowed category values.<br/>**Branch link:** https://github.com/BI-course/smm-202604-a6/tree/feature/arm-music-recommender<br/>**What I learned:** The rules are mined once in the notebook and stored as a lookup, so the API only does a fast subset match and ranks the results by confidence, the same way a production recommender microservice works. An unknown track ID should return a 400 that names the bad ID, and a discovery endpoint lets developers integrate without reading our code. |

**Member 5:**

| **Details**                                                                                                                           | **Comment** |
|:--------------------------------------------------------------------------------------------------------------------------------------|:------------|
| **Student ID**                                                                                                                        | 167746 |
| **Name**                                                                                                                              | Turkmen Desmond |
| **What part of the lab did you personally<br/>contribute to (provide a link to the<br/>branch(es)), and what did you learn from it?** | **Part:** Part C, Deployment. Containerised the unmodified `api.py` with Gunicorn behind an Nginx reverse proxy (Docker Compose, self-signed TLS, HTTP to HTTPS redirect). Deployed it to Render with a start command sized for the free tier. Published the Gradio app as a Hugging Face Space (Python 3.12 pinned in the Space's README) and the Streamlit app on Streamlit Community Cloud, all using the same `.joblib` artifacts in `model/`.<br/>**Branch link:** https://github.com/BI-course/smm-202604-a6/tree/feature/deployment-docker-render<br/>**What I learned:** Nginx terminates HTTPS and forwards `/api/` to Gunicorn's worker processes, which replaces Flask's single-threaded development server. Each Gunicorn worker loads every model (about 110 MB), so Render's 512 MB free tier needs 2 workers, not 4. A pickled model only works reliably with the scikit-learn version that saved it, so every platform pins `scikit-learn==1.9.0` and Python 3.12. |

## Chosen Level of Difficulty

**Specify the chosen level of difficulty** (baseline, intermediate, or advanced): **Advanced**

## Video Demonstration

Submit the link to a short video (**not more than 10 minutes**) demonstrating your complete solution at whichever difficulty level you chose. Please ensure that the lecturer has rights to view the video.

At minimum, the video must show, in one continuous walkthrough:

1. `api.py` running **locally**, with a live request/response against each model endpoint you implemented.
2. If you attempted the Intermediate or Advanced tier: the same API running inside its **Gunicorn + Nginx Docker container**, reached through the reverse proxy, not the Flask development server.
3. If you attempted the Advanced tier: your **Hugging Face Space** and your **Streamlit Community Cloud app**, each opened live in a browser, with at least one prediction made on each tab, and your **Render**-hosted API answering a request from Postman or `curl`.

**Why this matters for grading:** free-tier public deployments are not guaranteed to still exist by the time this is graded. The video is treated as the authoritative record of what was actually working at submission time; a broken link at grading time is not penalized if the video clearly shows that deployment working.

Note that you are required to submit the link to the video and NOT the video itself. The video should NOT be uploaded to your repository—that would be a misuse of GitHub.

**Link to the video:**

## Public URLs

Fill in only the rows that apply to your chosen difficulty level (see Part C below). Leave a row blank if your tier does not require that deployment target.

| **Deployment target**                             | **URL** |
|:--------------------------------------------------|:--------|
| Render (the full `api.py`, all endpoints)         |         |
| Hugging Face Space (Gradio, one tab per model)    |         |
| Streamlit Community Cloud app (one tab per model) |         |

---

## Scenario

Across earlier labs, **your group** was assigned its own dataset and trained your own models for four distinct tasks:

1. A **regression** task
2. A **classification** task
3. A **k-Means clustering** task
4. An **Apriori association-rule mining** task

Different groups worked with different datasets and, in some cases, different algorithms for each task -- there is no single shared reference repository or model list for the whole class. Wherever this document refers to "your regressor," "your classifier," "your clustering model," or "your recommender," it means **whichever model your own group already trained and saved for that task**, not a specific named algorithm.

You are now required to build a single Flask API (`api.py`) that serves **your own group's four models**, loaded from disk (never retrained inside `api.py`), and to demonstrate that API running correctly across a growing number of deployment surfaces depending on the difficulty tier you choose.

### Part A: Regression and Classification

- Locate the trained regression and classification model files your group produced and saved in your earlier labs on Regression and Classification.
- Update [api.py](api.py) to include end-points that load these two models from disk and serve predictions from them.

### Part B: Clustering and Association Rule Mining

- Locate the trained clustering model and the association rules your group produced and saved in your earlier labs on Clustering and Association Rule Mining.
- Update [api.py](api.py) to include end-points that:
  - Load the clustering model from disk and predict which cluster a new client belongs to
  - Load the association rules from disk and recommend products based on them

**Note 1:** **`api.py` is NOT production-grade as it is.** It is only meant for demonstration purposes. Scalability and security must be taken into consideration before deploying an API in a production environment.

**Note 2:** Some students often treat the API as an afterthought, focusing only on training ML models. In practice, **the API is the product** -- it is how others interact with your model. The "hidden" learning here is that the delivery mechanism (API design, usability, error handling, and even documentation) often matters more to stakeholders in the industry than the models themselves.

### Part C: Deployment Breadth (Read Before Choosing Your Tier)

All three tiers below ask you to serve **the same four models** (your regressor, your classifier, your clustering model, your recommender) from **one, unmodified `api.py`**. The model files and the Flask routes do not change between tiers -- only how many different places that one working API is shown to run correctly.

**Baseline (Required):**

- Update `api.py` to serve **at least three of your four** models, loaded from disk.
- Run and demonstrate `api.py` **locally** with Flask's development server, with a successful request against each implemented endpoint.

**Intermediate (Recommended):**

- Update `api.py` to serve **all four** of your models, loaded from disk.
- Dockerize your Flask API using **Gunicorn**, and place it behind an **Nginx** reverse proxy (see the reverse-proxy lab notes for the full Docker Compose setup).
- Deploy the same, unmodified `api.py` to **Render**, and demonstrate a real request against it from outside your own machine (cURL or Postman).

**Advanced (Optional):**

- Everything required for Intermediate, plus:
  - Create a web page (or pages) using Basic HTML, CSS, and Vanilla JavaScript that demonstrates the use of the API
  - Implement basic error handling (e.g., missing inputs)
  - Build **one Gradio app with a tab per model** (all four of your models) and publish it as a **Hugging Face Space**
  - Build **one Streamlit app with a tab per model**, covering the same four models, and publish it on **Streamlit Community Cloud**
  - All deployment targets -- local, Dockerized/Nginx, Render, Hugging Face, and Streamlit -- must serve models loaded **from the same trained artifacts**; retraining different models per platform does not satisfy this requirement

Why this is production-relevant beyond the model itself:

- **Portability:** the same container image and the same trained artifacts run unmodified on your laptop, inside Docker, and on a managed cloud host.
- **Consistency:** no "but it works on my machine" gap between what you demo locally and what a domain expert sees on a public URL.
- **Separation of concerns:** a REST API (Render), a no-code demo UI (Hugging Face/Streamlit), and your own frontend (the Advanced-tier HTML pages) are three different ways of exposing the *same* underlying models to three different kinds of user -- a developer integrating your API, a non-technical domain expert validating it, and an end customer.

---

## Marking Guide

Refer to the marking guide available here: [lab_submission_marking_guide.md](lab_submission_marking_guide.md)