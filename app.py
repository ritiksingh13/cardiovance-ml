import logging
import math
import os
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "heart_disease_bundle.pkl"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("heart-risk")

_bundle = joblib.load(MODEL_PATH)
MODEL = _bundle["model"]
SCALER = _bundle["scaler"]
FEATURES = list(_bundle["feature_columns"])
NUMERIC = list(_bundle["numerical_columns"])
CHOL_MEAN = float(_bundle["cholesterol_mean"])
BP_MEAN = float(_bundle["restingbp_mean"])

NUMERIC_RULES = {
    "Age": (1, 120, "Age"),
    "RestingBP": (50, 250, "Resting blood pressure"),
    "Cholesterol": (80, 700, "Total cholesterol"),
    "MaxHR": (50, 250, "Maximum heart rate"),
    "Oldpeak": (-3, 7, "ST depression"),
}
CHOICES = {
    "Gender": {"M", "F"},
    "ChestPainType": {"TA", "ATA", "NAP", "ASY"},
    "FastingBS": {"0", "1"},
    "RestingECG": {"Normal", "ST", "LVH"},
    "ExerciseAngina": {"Y", "N"},
    "ST_Slope": {"Up", "Flat", "Down"},
}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024


@app.after_request
def security_headers(resp):
    resp.headers["X-Content-Type-Options"] = "nosniff"
    resp.headers["X-Frame-Options"] = "DENY"
    resp.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    if request.path.startswith("/api/"):
        resp.headers["Cache-Control"] = "no-store"
    return resp


def validate(data):
    """Return (clean_values, errors)."""
    clean, errors = {}, {}
    if not isinstance(data, dict):
        return clean, {"_": "Send a JSON object."}

    for name, (lo, hi, label) in NUMERIC_RULES.items():
        raw = data.get(name)
        if raw is None or isinstance(raw, bool) or str(raw).strip() == "":
            errors[name] = f"{label} is required."
            continue
        try:
            val = float(raw)
        except (TypeError, ValueError):
            errors[name] = f"{label} must be a number."
            continue
        if not math.isfinite(val):
            errors[name] = f"{label} must be a number."
        elif name == "Cholesterol" and val == 0:
            clean[name] = 0.0
        elif not lo <= val <= hi:
            errors[name] = f"{label} must be between {lo} and {hi}."
        else:
            clean[name] = val

    for name, allowed in CHOICES.items():
        val = str(data.get(name, "")).strip()
        if val not in allowed:
            errors[name] = f"{name} must be one of: {', '.join(sorted(allowed))}."
        else:
            clean[name] = val
    return clean, errors


def build_row(v):
    """Rebuild the exact feature layout used in training (get_dummies, drop_first=True)."""
    row = {c: 0.0 for c in FEATURES}
    chol_missing = v["Cholesterol"] == 0
    row.update(
        Age=v["Age"],
        RestingBP=v["RestingBP"] if v["RestingBP"] > 0 else BP_MEAN,
        Cholesterol=CHOL_MEAN if chol_missing else v["Cholesterol"],
        FastingBS=float(v["FastingBS"]),
        MaxHR=v["MaxHR"],
        Oldpeak=v["Oldpeak"],
    )
    one_hot = {
        "Gender_M": v["Gender"] == "M",
        "ExerciseAngina_Y": v["ExerciseAngina"] == "Y",
        f"ChestPainType_{v['ChestPainType']}": True,
        f"RestingECG_{v['RestingECG']}": True,
        f"ST_Slope_{v['ST_Slope']}": True,
    }
    for col, on in one_hot.items():
        if col in row:
            row[col] = 1.0 if on else 0.0
    df = pd.DataFrame([row], columns=FEATURES)
    df[NUMERIC] = SCALER.transform(df[NUMERIC].astype(float))
    return df, chol_missing


@app.get("/")
def index():
    return render_template("index.html", active="predict")


@app.get("/guide")
def guide():
    return render_template("guide.html", active="guide")


@app.get("/health")
def health():
    return jsonify(status="ok")


@app.post("/api/predict")
def predict():
    values, errors = validate(request.get_json(silent=True))
    if errors:
        return jsonify(error="Please check your inputs.", fields=errors), 400
    try:
        X, chol_missing = build_row(values)
        probability = float(MODEL.predict_proba(X)[0][1])
        prediction = int(MODEL.predict(X)[0])
    except Exception:
        log.exception("Prediction failed")
        return jsonify(error="Something went wrong while making the prediction."), 500
    return jsonify(
        prediction=prediction,
        probability=round(probability, 4),
        cholesterol_estimated=chol_missing,
    )


@app.errorhandler(404)
def not_found(_):
    if request.path.startswith("/api/"):
        return jsonify(error="Not found."), 404
    return render_template("base.html", active="", not_found=True), 404


@app.errorhandler(405)
def bad_method(_):
    return jsonify(error="Method not allowed."), 405


@app.errorhandler(413)
def too_large(_):
    return jsonify(error="Request too large."), 413


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)