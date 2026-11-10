"""
Hugging Face Space (Gradio) -- one app, one tab per model, serving OUR GROUP'S
four models from the SAME trained .joblib artifacts that api.py serves:

    1. Regression       -- Nairobi monthly rental price      (Lasso, log1p target)

Nothing is retrained here: every tab only calls .predict()/.transform() on, or
looks up, the artifacts loaded once at start-up.

Files to upload to the Space (all in the Space's root folder):
    app.py
    requirements.txt
    lasso_regressor_for_rental_price.joblib

Run locally from the repository root (it then finds the files in ./model):
    python huggingface-spaces-using-gradio/app.py
"""

import os

import gradio as gr
import joblib
import numpy as np
import pandas as pd

# -----------------------------------------------------------------------
# Hugging Face's free tier provisions new Gradio Spaces on ZeroGPU hardware.
# None of these models needs a GPU, but some free-tier Spaces fail to start
# unless at least one function is decorated with @spaces.GPU. This no-op only
# satisfies that platform check. The `spaces` package exists only on Hugging
# Face, so locally the decorator is skipped.
# -----------------------------------------------------------------------
try:
    import spaces

    @spaces.GPU
    def _warm_zerogpu():
        return None

    _warm_zerogpu()
except ImportError:
    pass


# -----------------------------------------------------------------------
# Load the four artifacts ONCE, at start-up (never inside a function).
# On Hugging Face they sit next to app.py; locally they are in ../model.
# -----------------------------------------------------------------------
APP_DIR = os.path.dirname(os.path.abspath(__file__))
SEARCH_DIRS = [APP_DIR, os.path.join(APP_DIR, "model"), os.path.join(APP_DIR, "..", "model")]


def load_artifact(filename):
    for folder in SEARCH_DIRS:
        path = os.path.join(folder, filename)
        if os.path.exists(path):
            return joblib.load(path)
    raise FileNotFoundError(f"{filename} not found in any of: {SEARCH_DIRS}")


rental_pipeline = load_artifact("lasso_regressor_for_rental_price.joblib")


def fitted_categories(pipeline, transformer_name, encoder_step):
    """The category values each pipeline was trained on -- used for dropdowns,
    so a user can never pick a value the model has not seen."""
    branch = pipeline.named_steps["preprocessor"].named_transformers_[transformer_name]
    encoder = branch.named_steps[encoder_step]
    return {col: [str(v) for v in cats] for col, cats in zip(branch.feature_names_in_, encoder.categories_)}


RENTAL_FIELDS = list(rental_pipeline.feature_names_in_)
RENTAL_CATS = fitted_categories(rental_pipeline, "nominal", "onehot")


def require(values, names, optional=()):
    """Basic error handling: stop with a clear message if an input is empty."""
    missing = [n for n, v in zip(names, values) if v is None and n not in optional]
    if missing:
        raise gr.Error("Please fill in: " + ", ".join(m.replace("_", " ") for m in missing))


# -----------------------------------------------------------------------
# 1. Regression -- monthly rental price (KES)
# -----------------------------------------------------------------------
def predict_rent(*values):
    require(values, RENTAL_FIELDS, optional=("service_charge_kes", "security_rating"))
    input_row = pd.DataFrame([dict(zip(RENTAL_FIELDS, values))])
    rent_kes = float(np.expm1(rental_pipeline.predict(input_row)[0]))  # model predicts log1p(rent)
    return f"## KES {rent_kes:,.0f} per month\n\nModel: Lasso Regressor (alpha=0.001, log1p target)"


# -----------------------------------------------------------------------
# One gr.Blocks() app, four gr.Tab()s
# -----------------------------------------------------------------------
def number(label, value, minimum=0, maximum=None, step=None):
    return gr.Number(label=label, value=value, minimum=minimum, maximum=maximum, step=step)


with gr.Blocks(title="BBT 4206 Group Model Demos") as demo:
    gr.Markdown("# BBT 4206: Our Group's Models\nOne tab per model. Every tab uses the same saved "
                "`.joblib` artifacts as our Flask API; nothing is retrained.")

    with gr.Tab("Rental Price (Regression)"):
        gr.Interface(
            fn=predict_rent,
            inputs=[{
                "num_bedrooms": number("Bedrooms", 3, step=1),
                "size_sqm": number("Size (sqm)", 120.0),
                "service_charge_kes": number("Service charge (KES, blank if unknown)", 15000.0),
                "distance_to_cbd_km": number("Distance to CBD (km)", 8.0),
                "nearby_schools_count": number("Nearby schools", 3, step=1),
                "amenities_score": number("Amenities score (0-100)", 65.0, maximum=100),
                "property_age_years": number("Property age (years)", 5.0),
                "security_rating": number("Security rating (1-5, blank if unknown)", 4.0, minimum=1, maximum=5),
                "property_type": gr.Dropdown(RENTAL_CATS["property_type"], value="Apartment", label="Property type"),
                "neighborhood": gr.Dropdown(RENTAL_CATS["neighborhood"], value="Kilimani", label="Neighborhood"),
                "furnishing_status": gr.Dropdown(RENTAL_CATS["furnishing_status"], value="Semi-Furnished", label="Furnishing"),
            }[f] for f in RENTAL_FIELDS],
            outputs=gr.Markdown(),
            flagging_mode="never",
        )

if __name__ == "__main__":
    demo.launch()
