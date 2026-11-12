# Cardiovance ML

Cardiovance ML is a Flask-based heart disease risk assessment web app that predicts the likelihood of heart disease using a trained machine learning model. It allows users to enter clinical and lifestyle indicators, then receives a risk prediction with a probability score.

This project combines a Python backend, a machine learning model, and a responsive web interface to make cardiovascular risk estimation more accessible and easy to understand.

## Features

- Interactive heart risk calculator
- 11-question clinical assessment form
- Probability-based prediction output
- Educational heart health guide page
- Input validation for medical values
- Deployment-ready Flask app for Render
- Model bundle generated from trained heart disease data

## Tech Stack

- Python
- Flask
- scikit-learn
- pandas
- NumPy
- joblib
- HTML / CSS / JavaScript

## Project Structure

```text
cardiovance-ml/
├── app.py                     # Flask app and prediction logic
├── requirements.txt          # Python dependencies
├── render.yaml               # Render deployment config
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

## How It Works

The app loads a pre-trained model bundle from `model/heart_disease_bundle.pkl` and uses it to predict whether a patient has a higher or lower likelihood of heart disease based on inputs such as:

- Age
- Gender
- Resting blood pressure
- Cholesterol
- Fasting blood sugar
- Resting ECG
- Exercise angina
- Max heart rate
- ST depression
- ST slope
- Chest pain type

The prediction endpoint validates user input and prepares the feature set to match the training pipeline before sending it to the model.

## Local Setup

### Prerequisites

- Python 3.10+
- pip

### Installation

```bash
git clone https://github.com/ritiksingh13/cardiovance-ml.git
cd cardiovance-ml
python -m venv .venv
source .venv/bin/activate    # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Run the App

```bash
python app.py
```

Then open:

```text
http://localhost:5000
```

## API Endpoint

### Predict heart risk

Endpoint:

```http
POST /api/predict
```

Example request body:

```json
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

```json
{
  "prediction": 1,
  "probability": 0.7423,
  "cholesterol_estimated": false
}
```

## Deployment

This project includes a `render.yaml` file for deployment on Render.

```yaml
services:
  - type: web
    name: heart-risk-check
    runtime: python
    buildCommand: pip install --upgrade pip && pip install -r requirements.txt
    startCommand: gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --threads 4 --timeout 60
    healthCheckPath: /health
```

## Notes

- The model is trained using a clinical heart disease dataset and packaged as a reusable prediction bundle.
- This app is intended for educational and informational use only.
- It is not a substitute for professional medical advice or diagnosis.

## License

This project does not currently include a license file. If you plan to distribute or publish it publicly, consider adding an appropriate open-source license.

## Author

Ritik Singh

## Acknowledgements

- Flask documentation
- scikit-learn machine learning ecosystem
- Heart disease dataset used for model training

