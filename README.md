# Cardiovance ML

Cardiovance ML is a Flask-based heart disease risk assessment web app that predicts the likelihood of heart disease using a trained machine learning model. It allows users to enter clinical and lifestyle indicators, then receive a risk prediction with a probability score.

This project combines a Python backend, a machine learning model, and a responsive web interface to make cardiovascular risk estimation more accessible and easy to understand.

**Live demo:** [https://cardiovance.onrender.com/](https://cardiovance.onrender.com/)

## Table of Contents

- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Model Training](#model-training)
- [How the App Uses the Model](#how-the-app-uses-the-model)
- [Local Setup](#local-setup)
- [Retraining the Model](#retraining-the-model)
- [API Endpoint](#api-endpoint)
- [Deployment](#deployment)
- [Limitations](#limitations)
- [Author](#author)

## Features

- Interactive heart risk calculator
- 11-question clinical assessment form
- Probability-based prediction output
- Educational heart health guide page
- Input validation for medical values
- Deployment-ready Flask app for Render
- Model bundle generated from the training notebook

## Tech Stack

- Python
- Flask
- scikit-learn
- pandas
- NumPy
- joblib
- Matplotlib and Seaborn (notebook visualizations)
- HTML / CSS / JavaScript

## Project Structure

```
cardiovance-ml/
├── app.py                     # Flask app and prediction logic
├── requirements.txt           # Python dependencies
├── render.yaml                # Render deployment config
├── .gitignore
├── model/
│   └── heart_disease_bundle.pkl
├── notebooks/
│   ├── heart.csv
│   └── heart_disease_model_training.ipynb
├── static/
│   ├── css/
│   └── js/
├── templates/
│   ├── base.html
│   ├── guide.html
│   └── index.html
└── README.md
```

## Model Training

All model development is done in `notebooks/heart_disease_model_training.ipynb`. The notebook follows a standard supervised classification workflow: load data, explore it, clean it, encode it, split it, scale it, train a model, evaluate it, and save everything needed for deployment.

### 1. Dataset

The training data is `notebooks/heart.csv`.

| Property | Value |
| --- | --- |
| Rows | 918 |
| Columns | 12 (11 input features and 1 target) |
| Target | `HeartDisease` (1 = heart disease, 0 = normal) |
| Positive class share | about 55% (mean of target is 0.553) |
| Missing values | None |
| Duplicate rows | None |

Input features:

| Feature | Type | Description |
| --- | --- | --- |
| `Age` | Numerical | Age in years |
| `Gender` | Categorical | M or F (renamed from the original `Sex` column) |
| `ChestPainType` | Categorical | ATA, NAP, ASY, TA |
| `RestingBP` | Numerical | Resting blood pressure |
| `Cholesterol` | Numerical | Serum cholesterol |
| `FastingBS` | Binary | Fasting blood sugar flag (0 or 1) |
| `RestingECG` | Categorical | Normal, ST, LVH |
| `MaxHR` | Numerical | Maximum heart rate achieved |
| `ExerciseAngina` | Categorical | Y or N |
| `Oldpeak` | Numerical | ST depression |
| `ST_Slope` | Categorical | Up, Flat, Down |

### 2. Exploratory Data Analysis

The notebook first checks the shape, data types, null counts, duplicate rows, and summary statistics with `df.describe()`. Then it visualizes the data:

- Histograms with KDE curves for `Age`, `RestingBP`, `Cholesterol`, and `MaxHR`
- Count plots of `Gender`, `ChestPainType`, `RestingECG`, `ExerciseAngina`, `ST_Slope`, and `FastingBS`, each split by `HeartDisease`
- A correlation heatmap of the numerical columns

### 3. Data Cleaning

The dataset has no null values, but the histograms and summary statistics revealed invalid zeros that act as hidden missing values:

| Column | Zero values found | Replacement |
| --- | --- | --- |
| `Cholesterol` | 172 | Mean of the non-zero values (244.64) |
| `RestingBP` | 1 | Mean of the non-zero values (132.54) |

A cholesterol of 0 or a resting blood pressure of 0 is not physiologically possible, so these were treated as missing entries and imputed with the mean of the valid values. Both means are stored in the model bundle so the web app can apply the same imputation to user input.

### 4. Encoding

Categorical columns are converted with `pd.get_dummies(df, drop_first=True)`, and the encoded dataframe is cast to integer type. This produces 15 model features:

- Numerical: `Age`, `RestingBP`, `Cholesterol`, `FastingBS`, `MaxHR`, `Oldpeak`
- `Gender_M`
- `ChestPainType_ATA`, `ChestPainType_NAP`, `ChestPainType_TA`
- `RestingECG_Normal`, `RestingECG_ST`
- `ExerciseAngina_Y`
- `ST_Slope_Flat`, `ST_Slope_Up`

### 5. Train and Test Split

| Setting | Value |
| --- | --- |
| Split | 80% train (734 rows), 20% test (184 rows) |
| `random_state` | 42 |
| `stratify` | Target column, so both sets keep the same class ratio |

### 6. Feature Scaling

`StandardScaler` is fitted on the training set only and then applied to the test set, which prevents information from the test data leaking into training. Scaling is applied to these columns:

`Age`, `RestingBP`, `Cholesterol`, `MaxHR`, `Oldpeak`

### 7. Model

A Logistic Regression classifier from scikit-learn, trained with `max_iter=1000`.

### 8. Results

Evaluated on the held-out test set of 184 patients:

| Metric | Value |
| --- | --- |
| Accuracy | 87.5% |

| Class | Precision | Recall | F1-score | Support |
| --- | --- | --- | --- | --- |
| 0 (no heart disease) | 0.86 | 0.85 | 0.86 | 82 |
| 1 (heart disease) | 0.88 | 0.89 | 0.89 | 102 |

Confusion matrix:

| | Predicted 0 | Predicted 1 |
| --- | --- | --- |
| **Actual 0** | 70 | 12 |
| **Actual 1** | 11 | 91 |

The model correctly identifies 91 of the 102 patients with heart disease, and 70 of the 82 patients without it.

### 9. Model Bundle

To avoid preprocessing mismatches between training and deployment, everything the app needs is saved together in one file, `heart_disease_bundle.pkl`, using `joblib`:

| Key | Contents |
| --- | --- |
| `model` | Trained Logistic Regression model |
| `scaler` | Fitted `StandardScaler` |
| `feature_columns` | Exact column order the model expects |
| `numerical_columns` | Columns that must be scaled |
| `cholesterol_mean` | Mean used to replace zero cholesterol |
| `restingbp_mean` | Mean used to replace zero resting blood pressure |

## How the App Uses the Model

`app.py` loads `model/heart_disease_bundle.pkl` at startup. For each request it:

1. Validates the submitted values.
2. Replaces missing or zero cholesterol and resting blood pressure with the stored training means.
3. One-hot encodes the categorical inputs and arranges the columns in the order given by `feature_columns`.
4. Scales the numerical columns with the stored `scaler`.
5. Passes the result to the model and returns the predicted class and probability.

## Local Setup

### Prerequisites

- Python 3.10+
- pip

### Installation

```
git clone https://github.com/ritiksingh13/cardiovance-ml.git
cd cardiovance-ml
python -m venv .venv
source .venv/bin/activate    # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Run the App

```
python app.py
```

Then open:

```
http://localhost:5000
```

## Retraining the Model

1. Install the notebook dependencies if needed: `pip install jupyter matplotlib seaborn`
2. Open `notebooks/heart_disease_model_training.ipynb` from inside the `notebooks` folder, because it reads `heart.csv` using a relative path.
3. Run all cells from top to bottom.
4. The last cell writes `heart_disease_bundle.pkl` to the current folder. Move it to `model/` to replace the existing bundle.

## API Endpoint

### Predict heart risk

Endpoint:

```
POST /api/predict
```

Example request body:

```
{
  "Age": 45,
  "Gender": "M",
  "ChestPainType": "ATA",
  "RestingBP": 130,
  "Cholesterol": 210,
  "FastingBS": "0",
  "RestingECG": "Normal",
  "MaxHR": 160,
  "ExerciseAngina": "N",
  "Oldpeak": 1.2,
  "ST_Slope": "Up"
}
```

Example response:

```
{
  "prediction": 1,
  "probability": 0.7423,
  "cholesterol_estimated": false
}
```

## Deployment

This project includes a `render.yaml` file for deployment on Render.

```
services:
  - type: web
    name: heart-risk-check
    runtime: python
    buildCommand: pip install --upgrade pip && pip install -r requirements.txt
    startCommand: gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --threads 4 --timeout 60
    healthCheckPath: /health
```

## Limitations

- The model is trained on a small dataset of 918 records, so it should not be treated as clinically validated.
- Only one algorithm (Logistic Regression) was trained and evaluated, and results come from a single train/test split.
- Zero values in cholesterol and resting blood pressure were replaced with the mean, which is a simple imputation approach.
- This app is intended for educational and informational use only. It is not a substitute for professional medical advice or diagnosis.

## License

This project does not currently include a license file. If you plan to distribute or publish it publicly, consider adding an appropriate open-source license.

## Author

Ritik Singh

## Acknowledgements

- Flask documentation
- scikit-learn machine learning ecosystem
- Heart disease dataset used for model training
