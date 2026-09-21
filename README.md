# Aegis HR - Intelligent Workforce Attrition & Retention Platform

A production-grade, end-to-end Machine Learning platform built for predicting employee attrition, identifying workforce flight risks, evaluating policy interventions through "What-If" simulations, and serving prescriptive retention recommendations.

---

## 🌟 Key Features

### 1. Advanced Machine Learning Pipeline
- **Multiple Model Benchmarking**: Trains and compares **Logistic Regression** (calibrated baseline), **Random Forest** (ensemble bagging with feature importances), and **Gradient Boosting** (sequential boosting).
- **Class Imbalance Optimization**: Employs balanced class weighting to handle the ~16% positive attrition distribution.
- **Cross-Validation**: 5-Fold Stratified Cross-Validation ensuring model generalization.
- **Evaluation Suite**: Calculates Accuracy, ROC-AUC, PR-AUC, F1-Score, Precision, Recall, and Confusion Matrices.
- **Artifact Persistence**: Automatically serializes trained models, scalers, encoders, and metrics into `backend/models/`.

### 2. Modern Glassmorphic Frontend Dashboard
- **Executive Overview**: Real-time KPIs (Total Headcount, Attrition Rate, Churn Count, Average Salary) and interactive breakdown charts.
- **Individual Risk Calculator**: Demographic and job role risk assessment form with quick presets, animated SVG radial risk gauge, top risk drivers, and prescriptive retention recommendations.
- **"What-If" Retention Simulator**: Interactive policy levers (Salary adjustment, Overtime elimination, Work-Life Balance score) that calculate live risk reduction deltas ($\Delta P$).
- **Batch CSV Risk Scanner**: Drag-and-drop CSV ingestion, automated vector inference, risk tier badges, search/filtering, and enriched CSV export.
- **Model Lab & Retraining Hub**: Side-by-side performance cards, confusion matrix visualizers, feature importance charts, and one-click retraining.

### 3. Production REST API
- Built with lightweight, high-performance Flask architecture with CORS enabled and standardized JSON responses.

### 4. Automated Testing Suite
- Comprehensive suite of 21 unit and integration tests covering data validation, preprocessing, model training, predictor inference, edge cases, and API endpoints.

---

## 📁 Repository Structure

```
ML/
├── backend/
│   ├── config.py                 # Hyperparameters, column schemas, paths, thresholds
│   ├── app.py                    # Flask REST API server & static asset host
│   ├── pipeline/
│   │   ├── data_loader.py        # Dataset loading, schema validation, missing data imputation
│   │   ├── preprocessor.py       # ColumnTransformer (StandardScaler + OneHotEncoder)
│   │   ├── trainer.py            # Model training, 5-Fold Stratified CV, model selection
│   │   ├── evaluator.py          # Metrics, confusion matrix, ROC data, feature importances
│   │   └── predictor.py          # Real-time inference, risk attribution, recommendation engine
│   └── models/                   # Serialized model artifacts (.joblib, metrics.json, metadata.json)
├── frontend/
│   ├── index.html                # Single Page Application layout
│   ├── css/
│   │   └── style.css             # Glassmorphic dark/light design system & animations
│   └── js/
│       ├── api.js                # Asynchronous API communication layer
│       ├── charts.js             # Dynamic chart visualizers (Chart.js + SVG fallback)
│       └── app.js                # State management, risk gauge, simulator & batch scanner
├── tests/
│   ├── test_data_loader.py       # Ingestion & validation tests
│   ├── test_preprocessor.py      # Scaling & encoding transformation tests
│   ├── test_trainer.py           # Model training & artifact persistence tests
│   ├── test_predictor.py         # Inference, batch scoring & recommendation tests
│   └── test_api.py               # Flask REST API integration tests
├── run_training.py               # Standalone CLI training script
├── run_tests.py                  # Standalone automated test runner
├── run_app.py                    # Application launch script (http://127.0.0.1:5000)
├── requirements.txt              # Project dependencies
└── WA_Fn-UseC_-HR-Employee-Attrition.csv # Dataset
```

---

## 🚀 Quickstart Guide

### 1. Run Automated Test Suite
To verify that all components, preprocessing pipelines, inference engines, and API endpoints are working correctly:
```bash
py -3.14 run_tests.py
```
*(Expected output: 21 passed tests with 0 failures)*

### 2. Train / Retrain Machine Learning Models
To run the full training pipeline with 5-fold cross-validation and print comparison benchmarks:
```bash
py -3.14 run_training.py
```

### 3. Launch the Web Application
To start the backend server and open the interactive dashboard:
```bash
py -3.14 run_app.py
```
Then open your browser to: **`http://127.0.0.1:5000`**

---

## 📡 REST API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/health` | Service health status and model readiness |
| `GET` | `/api/dataset/stats` | Dataset statistics and department distributions |
| `GET` | `/api/models/metrics` | Benchmark metrics, confusion matrices, feature importances |
| `POST` | `/api/models/train` | Triggers dynamic model retraining |
| `POST` | `/api/predict` | Single employee attrition risk and recommendations |
| `POST` | `/api/predict/batch` | Bulk employee prediction via CSV upload or JSON list |
| `POST` | `/api/simulate` | "What-If" policy intervention simulation |

### Sample Prediction Request (`POST /api/predict`)
```json
{
  "Age": 32,
  "BusinessTravel": "Travel_Rarely",
  "DailyRate": 800,
  "Department": "Sales",
  "DistanceFromHome": 15,
  "Education": 3,
  "EducationField": "Life Sciences",
  "EnvironmentSatisfaction": 2,
  "Gender": "Female",
  "HourlyRate": 70,
  "JobInvolvement": 2,
  "JobLevel": 2,
  "JobRole": "Sales Executive",
  "JobSatisfaction": 2,
  "MaritalStatus": "Single",
  "MonthlyIncome": 4500,
  "MonthlyRate": 15000,
  "NumCompaniesWorked": 3,
  "OverTime": "Yes",
  "PercentSalaryHike": 12,
  "PerformanceRating": 3,
  "RelationshipSatisfaction": 2,
  "StockOptionLevel": 0,
  "TotalWorkingYears": 7,
  "TrainingTimesLastYear": 2,
  "WorkLifeBalance": 2,
  "YearsAtCompany": 4,
  "YearsInCurrentRole": 2,
  "YearsSinceLastPromotion": 2,
  "YearsWithCurrManager": 2
}
```

### Sample Prediction Response
```json
{
  "success": true,
  "data": {
    "attrition_prediction": 1,
    "attrition_label": "Will Leave",
    "probability": 0.724,
    "risk_percentage": 72.4,
    "risk_level": "HIGH",
    "risk_label": "High Risk",
    "risk_color": "#f97316",
    "top_drivers": [
      { "feature": "OverTime", "description": "Frequent OverTime shifts detected", "impact": "negative" },
      { "feature": "JobSatisfaction", "description": "Low Job Satisfaction rating (2/4)", "impact": "negative" }
    ],
    "recommendations": [
      { "factor": "Frequent OverTime", "action": "Cap weekly overtime hours and redistribute workload.", "urgency": "High" },
      { "factor": "Low Job Satisfaction", "action": "Conduct a 1-on-1 skip-level meeting to align career interests.", "urgency": "High" }
    ]
  }
}
```
