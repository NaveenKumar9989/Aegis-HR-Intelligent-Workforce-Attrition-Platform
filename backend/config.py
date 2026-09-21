"""
Backend Configuration for Employee Attrition ML Platform.
Defines schemas, paths, hyperparameter grids, and operational constants.
"""

from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "WA_Fn-UseC_-HR-Employee-Attrition.csv"
MODELS_DIR = BASE_DIR / "backend" / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Target Column
TARGET_COL = "Attrition"
TARGET_MAPPING = {"No": 0, "Yes": 1}
TARGET_NAMES = ["Retained", "Attrited"]

# Columns that are non-informative or invariant
DROP_COLS = [
    "EmployeeCount",
    "EmployeeNumber",
    "Over18",
    "StandardHours"
]

# Categorical features
CATEGORICAL_COLS = [
    "BusinessTravel",
    "Department",
    "EducationField",
    "Gender",
    "JobRole",
    "MaritalStatus",
    "OverTime"
]

# Numerical features
NUMERICAL_COLS = [
    "Age",
    "DailyRate",
    "DistanceFromHome",
    "Education",
    "EnvironmentSatisfaction",
    "HourlyRate",
    "JobInvolvement",
    "JobLevel",
    "JobSatisfaction",
    "MonthlyIncome",
    "MonthlyRate",
    "NumCompaniesWorked",
    "PercentSalaryHike",
    "PerformanceRating",
    "RelationshipSatisfaction",
    "StockOptionLevel",
    "TotalWorkingYears",
    "TrainingTimesLastYear",
    "WorkLifeBalance",
    "YearsAtCompany",
    "YearsInCurrentRole",
    "YearsSinceLastPromotion",
    "YearsWithCurrManager"
]

# Risk Thresholds
RISK_LEVELS = {
    "LOW": {"max": 0.25, "label": "Low Risk", "color": "#10b981"},
    "MEDIUM": {"max": 0.50, "label": "Moderate Risk", "color": "#f59e0b"},
    "HIGH": {"max": 0.75, "label": "High Risk", "color": "#f97316"},
    "CRITICAL": {"max": 1.00, "label": "Critical Risk", "color": "#ef4444"}
}

# Artifact Filenames
BEST_MODEL_PATH = MODELS_DIR / "best_model.joblib"
PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.joblib"
METRICS_PATH = MODELS_DIR / "metrics.json"
METADATA_PATH = MODELS_DIR / "metadata.json"
ALL_MODELS_PATH = MODELS_DIR / "all_models.joblib"

RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5
