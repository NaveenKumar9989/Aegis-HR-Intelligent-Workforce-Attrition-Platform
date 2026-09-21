"""
Preprocessor & Feature Transformer for Employee Attrition.
Encapsulates scaling, one-hot encoding, feature name tracking, and artifact persistence.
"""

from typing import List, Optional
import joblib
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from backend.config import (
    CATEGORICAL_COLS,
    NUMERICAL_COLS,
    PREPROCESSOR_PATH,
)


class DataPreprocessor:
    """Preprocesses tabular data using StandardScaler and OneHotEncoder."""

    def __init__(self):
        self.numerical_cols = NUMERICAL_COLS
        self.categorical_cols = CATEGORICAL_COLS
        self.column_transformer: Optional[ColumnTransformer] = None
        self.feature_names_: List[str] = []
        self.is_fitted: bool = False

    def build_transformer(self) -> ColumnTransformer:
        """Constructs a ColumnTransformer with scaling and encoding."""
        numeric_pipeline = StandardScaler()
        categorical_pipeline = OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        )

        transformer = ColumnTransformer(
            transformers=[
                ("num", numeric_pipeline, self.numerical_cols),
                ("cat", categorical_pipeline, self.categorical_cols),
            ],
            remainder="drop"
        )
        return transformer

    def fit(self, X: pd.DataFrame) -> "DataPreprocessor":
        """Fits the transformers on the training dataframe."""
        self.column_transformer = self.build_transformer()
        self.column_transformer.fit(X)

        # Extract generated feature names
        num_features = self.numerical_cols
        cat_encoder = self.column_transformer.named_transformers_["cat"]
        cat_features = cat_encoder.get_feature_names_out(self.categorical_cols).tolist()
        self.feature_names_ = list(num_features) + list(cat_features)
        self.is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transforms feature dataframe into scaled/encoded numpy array."""
        if not self.is_fitted or self.column_transformer is None:
            raise RuntimeError("Preprocessor has not been fitted or loaded yet.")
        return self.column_transformer.transform(X)

    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        """Fits and transforms in a single call."""
        return self.fit(X).transform(X)

    def save(self, file_path=None) -> str:
        """Serializes the preprocessor artifact to disk."""
        target_path = file_path or PREPROCESSOR_PATH
        joblib.dump(
            {
                "transformer": self.column_transformer,
                "feature_names": self.feature_names_,
                "numerical_cols": self.numerical_cols,
                "categorical_cols": self.categorical_cols,
                "is_fitted": self.is_fitted
            },
            target_path
        )
        return str(target_path)

    @classmethod
    def load(cls, file_path=None) -> "DataPreprocessor":
        """Loads a persisted preprocessor from disk."""
        target_path = file_path or PREPROCESSOR_PATH
        if not target_path.exists():
            raise FileNotFoundError(f"Preprocessor artifact not found at {target_path}")

        data = joblib.load(target_path)
        instance = cls()
        instance.column_transformer = data["transformer"]
        instance.feature_names_ = data["feature_names"]
        instance.numerical_cols = data["numerical_cols"]
        instance.categorical_cols = data["categorical_cols"]
        instance.is_fitted = data["is_fitted"]
        return instance
