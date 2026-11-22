"""
Train the CreditWise loan approval model.

Expected input file:
    loan_approval_data.csv

The model follows the feature engineering used in the CreditWise notebook:
- Drops Applicant_ID
- Creates DTI_Ratio_sq and Credit_Score_sq
- Removes the original DTI_Ratio and Credit_Score columns
- Imputes missing values
- One-hot encodes categorical columns
- Scales features
- Trains Gaussian Naive Bayes

Run:
    python train_model.py
"""

from pathlib import Path
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "loan_approval_data.csv"
if not DATA_PATH.exists():
    downloaded_copy = BASE_DIR / "loan_approval_data (1).csv"
    if downloaded_copy.exists():
        DATA_PATH = downloaded_copy
MODEL_PATH = BASE_DIR / "loan_model.pkl"

TARGET = "Loan_Approved"

NUMERIC_FEATURES = [
    "Applicant_Income",
    "Coapplicant_Income",
    "Age",
    "Dependents",
    "Existing_Loans",
    "Savings",
    "Collateral_Value",
    "Loan_Amount",
    "Loan_Term",
    "DTI_Ratio_sq",
    "Credit_Score_sq",
]

CATEGORICAL_FEATURES = [
    "Employment_Status",
    "Marital_Status",
    "Loan_Purpose",
    "Property_Area",
    "Education_Level",
    "Gender",
    "Employer_Category",
]

RAW_REQUIRED_COLUMNS = [
    "Applicant_Income",
    "Coapplicant_Income",
    "Employment_Status",
    "Age",
    "Marital_Status",
    "Dependents",
    "Credit_Score",
    "Existing_Loans",
    "DTI_Ratio",
    "Savings",
    "Collateral_Value",
    "Loan_Amount",
    "Loan_Term",
    "Loan_Purpose",
    "Property_Area",
    "Education_Level",
    "Gender",
    "Employer_Category",
    TARGET,
]


def make_features(data: pd.DataFrame) -> pd.DataFrame:
    """Convert raw applicant columns into the feature format used by the model."""
    data = data.copy()

    missing = [column for column in RAW_REQUIRED_COLUMNS if column not in data.columns]
    if missing:
        raise ValueError(
            "The dataset is missing these required columns: " + ", ".join(missing)
        )

    data["DTI_Ratio_sq"] = pd.to_numeric(data["DTI_Ratio"], errors="coerce") ** 2
    data["Credit_Score_sq"] = pd.to_numeric(data["Credit_Score"], errors="coerce") ** 2

    # Match the feature engineering in the notebook.
    data = data.drop(columns=["Credit_Score", "DTI_Ratio", "Applicant_ID"], errors="ignore")

    return data[NUMERIC_FEATURES + CATEGORICAL_FEATURES]


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Could not find {DATA_PATH.name}. Place your dataset in this project folder."
        )

    df = pd.read_csv(DATA_PATH)

    missing = [column for column in RAW_REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(
            "Your CSV does not match the notebook's columns. Missing: "
            + ", ".join(missing)
        )

    # Convert target labels safely: Yes = 1, No = 0.
    target_text = df[TARGET].astype(str).str.strip().str.lower()
    target_map = {"yes": 1, "no": 0}
    y = target_text.map(target_map)

    valid_rows = y.notna()
    df = df.loc[valid_rows].copy()
    y = y.loc[valid_rows].astype(int)

    X = make_features(df)

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="mean")),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    drop="first",
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )

    # Scaling and classifier are kept inside one pipeline so Flask applies
    # exactly the same preprocessing that was learned during training.
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("scaler", StandardScaler()),
            ("classifier", GaussianNB()),
        ]
    )

    stratify = y if y.nunique() == 2 and y.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=stratify,
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print("\nCreditWise model evaluation")
    print("---------------------------")
    print(f"Accuracy: {accuracy_score(y_test, predictions):.4f}")
    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, predictions))
    print("\nClassification report:")
    print(classification_report(y_test, predictions, zero_division=0))

    joblib.dump(model, MODEL_PATH)
    print(f"\nTrained model saved successfully: {MODEL_PATH.name}")


if __name__ == "__main__":
    main()
