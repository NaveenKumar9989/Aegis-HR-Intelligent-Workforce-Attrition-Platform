"""
Data Loader & Validator for Employee Attrition Dataset.
Handles loading, schema validation, missing data imputation, and invariant feature cleanup.
"""

from typing import Tuple, Optional, Dict, Any
import pandas as pd
import numpy as np

from backend.config import (
    DATA_PATH,
    TARGET_COL,
    TARGET_MAPPING,
    DROP_COLS,
    CATEGORICAL_COLS,
    NUMERICAL_COLS,
)


class DataLoader:
    """Loads, validates, and cleans the HR Employee Attrition dataset."""

    def __init__(self, file_path: Optional[str] = None):
        self.file_path = file_path or DATA_PATH

    def load_raw_data(self) -> pd.DataFrame:
        """Load raw CSV data."""
        if not self.file_path or not pd.io.common.file_exists(str(self.file_path)):
            raise FileNotFoundError(f"Dataset file not found at: {self.file_path}")
        df = pd.read_csv(self.file_path)
        return df

    def validate_schema(self, df: pd.DataFrame, require_target: bool = True) -> bool:
        """
        Validate that the dataframe contains expected columns.
        Raises ValueError if mandatory columns are missing.
        """
        expected_features = set(CATEGORICAL_COLS + NUMERICAL_COLS)
        df_cols = set(df.columns)

        missing_features = expected_features - df_cols
        if missing_features:
            raise ValueError(f"Missing required feature columns: {sorted(list(missing_features))}")

        if require_target and TARGET_COL not in df.columns:
            raise ValueError(f"Target column '{TARGET_COL}' not found in dataset")

        return True

    def clean_and_prepare(
        self, df: pd.DataFrame, require_target: bool = True
    ) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """
        Cleans data:
        1. Drops uninformative/invariant columns
        2. Imputes any missing values with median/mode
        3. Encodes target 'Attrition' to binary (0/1)
        """
        self.validate_schema(df, require_target=require_target)
        df_clean = df.copy()

        # Drop invariant columns if present
        cols_to_drop = [c for c in DROP_COLS if c in df_clean.columns]
        if cols_to_drop:
            df_clean = df_clean.drop(columns=cols_to_drop)

        # Handle target if present
        y = None
        if TARGET_COL in df_clean.columns:
            # Map string to 0/1 if not already numeric
            if df_clean[TARGET_COL].dtype == object or isinstance(df_clean[TARGET_COL].iloc[0], str):
                df_clean[TARGET_COL] = df_clean[TARGET_COL].map(TARGET_MAPPING)
            y = df_clean[TARGET_COL].astype(int)
            df_clean = df_clean.drop(columns=[TARGET_COL])

        # Impute missing values defensively
        for col in NUMERICAL_COLS:
            if col in df_clean.columns:
                if df_clean[col].isnull().any():
                    median_val = df_clean[col].median()
                    df_clean[col] = df_clean[col].fillna(median_val)

        for col in CATEGORICAL_COLS:
            if col in df_clean.columns:
                if df_clean[col].isnull().any():
                    mode_val = df_clean[col].mode()[0] if not df_clean[col].mode().empty else "Unknown"
                    df_clean[col] = df_clean[col].fillna(mode_val)

        return df_clean, y

    def get_dataset_stats(self) -> Dict[str, Any]:
        """Compute exploratory analytics and summaries on the full raw dataset."""
        raw_df = self.load_raw_data()
        clean_df, y = self.clean_and_prepare(raw_df, require_target=True)

        total_employees = len(raw_df)
        attrition_count = int((raw_df[TARGET_COL] == "Yes").sum())
        retained_count = total_employees - attrition_count
        attrition_rate = round((attrition_count / total_employees) * 100, 2)

        # Department breakdown
        dept_stats = []
        for dept, group in raw_df.groupby("Department"):
            cnt = len(group)
            att = int((group[TARGET_COL] == "Yes").sum())
            rate = round((att / cnt) * 100, 1) if cnt > 0 else 0
            dept_stats.append({
                "department": dept,
                "total": cnt,
                "attrited": att,
                "retained": cnt - att,
                "attrition_rate": rate
            })

        # OverTime correlation
        ot_stats = []
        for ot, group in raw_df.groupby("OverTime"):
            cnt = len(group)
            att = int((group[TARGET_COL] == "Yes").sum())
            rate = round((att / cnt) * 100, 1) if cnt > 0 else 0
            ot_stats.append({
                "overtime": ot,
                "total": cnt,
                "attrited": att,
                "attrition_rate": rate
            })

        # Job Role breakdown
        role_stats = []
        for role, group in raw_df.groupby("JobRole"):
            cnt = len(group)
            att = int((group[TARGET_COL] == "Yes").sum())
            rate = round((att / cnt) * 100, 1) if cnt > 0 else 0
            role_stats.append({
                "role": role,
                "total": cnt,
                "attrited": att,
                "attrition_rate": rate,
                "avg_income": round(float(group["MonthlyIncome"].mean()), 0)
            })

        # Monthly income by attrition
        avg_income_attrited = round(float(raw_df[raw_df[TARGET_COL] == "Yes"]["MonthlyIncome"].mean()), 2)
        avg_income_retained = round(float(raw_df[raw_df[TARGET_COL] == "No"]["MonthlyIncome"].mean()), 2)

        return {
            "total_employees": total_employees,
            "attrition_count": attrition_count,
            "retained_count": retained_count,
            "attrition_rate": attrition_rate,
            "avg_income_attrited": avg_income_attrited,
            "avg_income_retained": avg_income_retained,
            "departments": dept_stats,
            "overtime": ot_stats,
            "roles": sorted(role_stats, key=lambda x: x["attrition_rate"], reverse=True)
        }
