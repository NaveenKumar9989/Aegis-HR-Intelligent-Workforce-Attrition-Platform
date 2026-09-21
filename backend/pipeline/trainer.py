"""
Model Training & Optimization Pipeline.
Trains, cross-validates, compares multiple models, and persists best artifacts.
"""

import json
from datetime import datetime
from typing import Dict, Any
import joblib
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from backend.config import (
    RANDOM_STATE,
    TEST_SIZE,
    CV_FOLDS,
    BEST_MODEL_PATH,
    PREPROCESSOR_PATH,
    METRICS_PATH,
    METADATA_PATH,
    ALL_MODELS_PATH,
)
from backend.pipeline.data_loader import DataLoader
from backend.pipeline.preprocessor import DataPreprocessor
from backend.pipeline.evaluator import ModelEvaluator


class ModelTrainer:
    """Trains, optimizes, and persists classification models."""

    def __init__(self, data_path: str = None):
        self.data_loader = DataLoader(data_path)
        self.preprocessor = DataPreprocessor()
        self.models: Dict[str, Any] = {}
        self.results: Dict[str, Any] = {}

    def get_candidate_models(self) -> Dict[str, Any]:
        """Define candidate classifiers optimized for imbalanced classification."""
        return {
            "Logistic Regression": LogisticRegression(
                class_weight="balanced",
                C=1.0,
                max_iter=2000,
                random_state=RANDOM_STATE
            ),
            "Random Forest": RandomForestClassifier(
                n_estimators=200,
                max_depth=12,
                min_samples_split=5,
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1
            ),
            "Gradient Boosting": GradientBoostingClassifier(
                n_estimators=150,
                learning_rate=0.08,
                max_depth=4,
                subsample=0.85,
                random_state=RANDOM_STATE
            )
        }

    def train_and_evaluate(self) -> Dict[str, Any]:
        """Executes full training, CV, benchmarking, and persistence workflow."""
        raw_df = self.data_loader.load_raw_data()
        X, y = self.data_loader.clean_and_prepare(raw_df, require_target=True)

        # Stratified train/test split to preserve ~16% positive class ratio
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=y
        )

        # Fit preprocessor strictly on training set
        X_train_trans = self.preprocessor.fit_transform(X_train)
        X_test_trans = self.preprocessor.transform(X_test)
        feature_names = self.preprocessor.feature_names_

        candidate_models = self.get_candidate_models()
        model_reports = {}
        cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

        best_score = -1.0
        best_model_name = ""
        best_model_obj = None

        for name, model in candidate_models.items():
            # 5-fold cross validation on training set
            cv_roc_auc = cross_val_score(model, X_train_trans, y_train, cv=cv, scoring="roc_auc")
            cv_f1 = cross_val_score(model, X_train_trans, y_train, cv=cv, scoring="f1")

            # Fit on full training set
            model.fit(X_train_trans, y_train)

            # Evaluate on unseen holdout test set
            eval_metrics = ModelEvaluator.evaluate(model, X_test_trans, y_test, feature_names=feature_names)
            eval_metrics["cv_roc_auc_mean"] = round(float(np.mean(cv_roc_auc)), 4)
            eval_metrics["cv_roc_auc_std"] = round(float(np.std(cv_roc_auc)), 4)
            eval_metrics["cv_f1_mean"] = round(float(np.mean(cv_f1)), 4)
            eval_metrics["cv_f1_std"] = round(float(np.std(cv_f1)), 4)

            model_reports[name] = eval_metrics
            self.models[name] = model

            # Composite ranking score: 70% ROC-AUC + 30% F1-Score
            rank_score = (0.7 * eval_metrics["roc_auc"]) + (0.3 * eval_metrics["f1_score"])
            if rank_score > best_score:
                best_score = rank_score
                best_model_name = name
                best_model_obj = model

        # Save artifacts
        self.preprocessor.save(PREPROCESSOR_PATH)
        joblib.dump(best_model_obj, BEST_MODEL_PATH)
        joblib.dump(self.models, ALL_MODELS_PATH)

        report = {
            "trained_at": datetime.utcnow().isoformat(),
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "total_features": len(feature_names),
            "best_model_name": best_model_name,
            "models": model_reports
        }

        # Save metrics JSON
        with open(METRICS_PATH, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        metadata = {
            "best_model_name": best_model_name,
            "trained_at": report["trained_at"],
            "metrics": model_reports[best_model_name],
            "feature_count": len(feature_names),
            "features": feature_names
        }
        with open(METADATA_PATH, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        self.results = report
        return report
