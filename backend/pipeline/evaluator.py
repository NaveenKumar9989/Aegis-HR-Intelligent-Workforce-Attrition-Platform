"""
Model Evaluator for Employee Attrition Classification.
Calculates classification metrics, confusion matrices, ROC/PR curves, and feature importance.
"""

from typing import Dict, Any, List
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    roc_curve,
)


class ModelEvaluator:
    """Evaluates classifier performance with a full suite of metrics."""

    @staticmethod
    def evaluate(model, X_test: np.ndarray, y_test: np.ndarray, feature_names: List[str] = None) -> Dict[str, Any]:
        """Runs full evaluation on test set."""
        y_pred = model.predict(X_test)
        has_proba = hasattr(model, "predict_proba")

        if has_proba:
            y_proba = model.predict_proba(X_test)[:, 1]
            roc_auc = float(roc_auc_score(y_test, y_proba))
            pr_auc = float(average_precision_score(y_test, y_proba))
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            # Sample curve points for frontend json serialization
            step = max(1, len(fpr) // 25)
            roc_curve_data = {
                "fpr": [round(float(x), 3) for x in fpr[::step]],
                "tpr": [round(float(y), 3) for y in tpr[::step]]
            }
        else:
            y_proba = y_pred
            roc_auc = 0.0
            pr_auc = 0.0
            roc_curve_data = {"fpr": [0.0, 1.0], "tpr": [0.0, 1.0]}

        acc = float(accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred, zero_division=0))
        rec = float(recall_score(y_test, y_pred, zero_division=0))
        f1 = float(f1_score(y_test, y_pred, zero_division=0))

        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = [int(v) for v in cm.ravel()]

        # Feature importances
        feature_importance_list = []
        if feature_names:
            if hasattr(model, "feature_importances_"):
                importances = model.feature_importances_
                indices = np.argsort(importances)[::-1]
                for idx in indices[:15]:
                    feature_importance_list.append({
                        "feature": feature_names[idx],
                        "importance": round(float(importances[idx]), 4)
                    })
            elif hasattr(model, "coef_"):
                coefs = np.abs(model.coef_[0])
                indices = np.argsort(coefs)[::-1]
                for idx in indices[:15]:
                    feature_importance_list.append({
                        "feature": feature_names[idx],
                        "importance": round(float(coefs[idx]), 4),
                        "direction": "positive" if model.coef_[0][idx] > 0 else "negative"
                    })

        report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

        return {
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "roc_auc": round(roc_auc, 4),
            "pr_auc": round(pr_auc, 4),
            "confusion_matrix": {
                "true_negative": tn,
                "false_positive": fp,
                "false_negative": fn,
                "true_positive": tp,
                "matrix": [[tn, fp], [fn, tp]]
            },
            "roc_curve": roc_curve_data,
            "feature_importances": feature_importance_list,
            "classification_report": report_dict
        }
