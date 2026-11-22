# CreditWise Flask + Machine Learning Project

This folder connects the CreditWise HTML/CSS frontend to a Flask backend and a scikit-learn loan approval model.

## 1. Project structure

```text
creditwise_flask_project/
├── app.py
├── train_model.py
├── loan_approval_data.csv   # You must copy your dataset here
├── loan_model.pkl           # Generated after model training
├── requirements.txt
├── templates/
│   └── index.html
└── static/
    └── style.css
```

## 2. Add your dataset

Copy your original `loan_approval_data.csv` into this project folder, beside `app.py`.

The CSV must contain the same columns as your notebook, including `Loan_Approved`. Do not rename the columns.

## 3. Create and activate a virtual environment (recommended)

Windows PowerShell:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, you can still use the environment's Python executable directly, or use Command Prompt:

```cmd
venv\Scripts\activate
```

## 4. Install libraries

```bash
pip install -r requirements.txt
```

## 5. Train the model

```bash
python train_model.py
```

This trains the Gaussian Naive Bayes model and creates `loan_model.pkl`. The script prints accuracy, confusion matrix, and classification report.

## 6. Start Flask

```bash
python app.py
```

Open this address in your browser:

```text
http://127.0.0.1:5000/
```

## How the integration works

1. `templates/index.html` collects the applicant's information.
2. The HTML form sends a POST request to `/predict`.
3. `app.py` reads the form values and creates the model input row.
4. The same feature engineering used during training is applied.
5. The saved scikit-learn pipeline performs imputation, encoding, scaling, and prediction.
6. Flask sends the result back to the HTML template.

## Important notes

- The HTML input names are aligned with the dataset columns used in the CreditWise notebook.
- The notebook's feature engineering is retained: `DTI_Ratio_sq` and `Credit_Score_sq` are created, while the original `DTI_Ratio` and `Credit_Score` are excluded from model input.
- The model is retrained from the CSV using a pipeline, so preprocessing is saved together with the classifier. This avoids manually reproducing encoders and scalers in Flask.
- The dataset is not included in this ZIP because it was not attached with the notebook/frontend files. Add your own `loan_approval_data.csv`.
- The result is for an academic demonstration only. It is not a real financial decision, and the displayed probability is not a calibrated lending probability.
- Do not deploy the Flask development server with debug mode on the public internet.
