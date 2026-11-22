from pathlib import Path
import math
from threading import Timer
import webbrowser

import joblib
import pandas as pd
from flask import Flask, flash, render_template, request

app = Flask(__name__)
app.secret_key = "change-this-to-a-random-secret-key"

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "loan_model.pkl"

# The trained model includes imputation, encoding and scaling.
model = None
if MODEL_PATH.exists():
    model = joblib.load(MODEL_PATH)

NUMERIC_FIELDS = [
    "Applicant_Income",
    "Coapplicant_Income",
    "Age",
    "Dependents",
    "Credit_Score",
    "Existing_Loans",
    "DTI_Ratio",
    "Savings",
    "Collateral_Value",
    "Loan_Amount",
    "Loan_Term",
]

CATEGORICAL_FIELDS = [
    "Employment_Status",
    "Marital_Status",
    "Loan_Purpose",
    "Property_Area",
    "Education_Level",
    "Gender",
    "Employer_Category",
]

ALLOWED_VALUES = {
    "Employment_Status": ["Salaried", "Self-employed", "Contract", "Unemployed"],
    "Marital_Status": ["Single", "Married"],
    "Loan_Purpose": ["Personal", "Car", "Business", "Education", "Home"],
    "Property_Area": ["Urban", "Semiurban", "Rural"],
    "Education_Level": ["Graduate", "Not Graduate"],
    "Gender": ["Male", "Female"],
    "Employer_Category": ["Private", "Government", "Business", "MNC", "Unemployed"],
}


def build_model_input(form):
    """Read form values and create one row with the same features as training."""
    values = {}

    for field in NUMERIC_FIELDS:
        raw_value = form.get(field, "").strip()
        if raw_value == "":
            raise ValueError(f"Please enter {field.replace('_', ' ')}.")
        try:
            values[field] = float(raw_value)
        except ValueError:
            raise ValueError(f"{field.replace('_', ' ')} must be a number.")
        if not math.isfinite(values[field]):
            raise ValueError(f"{field.replace('_', ' ')} must be a valid number.")

    for field in CATEGORICAL_FIELDS:
        value = form.get(field, "").strip()
        if not value:
            raise ValueError(f"Please select {field.replace('_', ' ')}.")
        if value not in ALLOWED_VALUES[field]:
            raise ValueError(f"Please select a valid {field.replace('_', ' ')}.")
        values[field] = value

    # Basic input validation.
    non_negative_fields = [
        "Applicant_Income", "Coapplicant_Income", "Age", "Dependents",
        "Credit_Score", "Existing_Loans", "Savings",
        "Collateral_Value", "Loan_Amount", "Loan_Term",
    ]
    for field in non_negative_fields:
        if values[field] < 0:
            raise ValueError(f"{field.replace('_', ' ')} cannot be negative.")

    if not 18 <= values["Age"] <= 100:
        raise ValueError("Age must be between 18 and 100.")
    if not 300 <= values["Credit_Score"] <= 900:
        raise ValueError("Credit Score must be between 300 and 900.")
    if not 0 <= values["Dependents"] <= 20:
        raise ValueError("Dependents must be between 0 and 20.")
    if not 0 <= values["Existing_Loans"] <= 50:
        raise ValueError("Existing Loans must be between 0 and 50.")
    if values["Loan_Term"] not in [12, 24, 36, 48, 60, 72, 84]:
        raise ValueError("Please select a valid loan term.")

    if not 0 <= values["DTI_Ratio"] <= 1:
        raise ValueError("DTI Ratio must be between 0 and 1 (for example, 0.35).")

    # Feature engineering must match train_model.py and the notebook.
    values["DTI_Ratio_sq"] = values["DTI_Ratio"] ** 2
    values["Credit_Score_sq"] = values["Credit_Score"] ** 2

    # The trained model was fitted without the original two columns.
    values.pop("DTI_Ratio")
    values.pop("Credit_Score")

    ordered_fields = [
        "Applicant_Income", "Coapplicant_Income", "Age", "Dependents",
        "Existing_Loans", "Savings", "Collateral_Value", "Loan_Amount",
        "Loan_Term", "DTI_Ratio_sq", "Credit_Score_sq",
        "Employment_Status", "Marital_Status", "Loan_Purpose",
        "Property_Area", "Education_Level", "Gender", "Employer_Category",
    ]

    return pd.DataFrame([[values[field] for field in ordered_fields]], columns=ordered_fields)


@app.route("/", methods=["GET"])
def home():
    if model is None:
        flash(
            "The trained model is not available yet. Add loan_approval_data.csv "
            "and run: python train_model.py"
        )
    return render_template("index.html", prediction=None, form_data=None)


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        flash("Model file not found. First run: python train_model.py")
        return render_template("index.html", prediction=None, form_data=request.form)

    try:
        applicant = build_model_input(request.form)
        predicted_class = int(model.predict(applicant)[0])

        # The training script maps No=0 and Yes=1.
        prediction = "Eligible (model predicts approval)" if predicted_class == 1 else "Not eligible (model predicts rejection)"

        probability = None
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(applicant)[0]
            class_positions = list(model.classes_)
            if 1 in class_positions:
                probability = round(float(probabilities[class_positions.index(1)]) * 100, 1)

        return render_template(
            "index.html",
            prediction=prediction,
            approved=predicted_class == 1,
            probability=probability,
            form_data=request.form,
        )

    except ValueError as error:
        flash(str(error))
        return render_template("index.html", prediction=None, form_data=request.form)


if __name__ == "__main__":
    # Local development only. Do not use debug=True on a public production server.
    Timer(1, lambda: webbrowser.open_new("http://127.0.0.1:5000/")).start()
    app.run(debug=False)
