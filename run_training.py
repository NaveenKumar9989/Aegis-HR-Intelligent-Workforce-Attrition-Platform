"""
Standalone CLI script to execute Model Training & Optimization Pipeline.
Usage:
    py -3.14 run_training.py
"""

import sys
import time
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from backend.pipeline.trainer import ModelTrainer


def print_banner():
    banner = """
========================================================================
   AEGIS HR - EMPLOYEE ATTRITION MACHINE LEARNING TRAINING PIPELINE     
========================================================================
    """
    print(banner)


def main():
    print_banner()
    t0 = time.time()
    print("[+] Initializing ModelTrainer...")
    trainer = ModelTrainer()

    print("[+] Loading dataset & running preprocessing...")
    print("[+] Training candidate models (Logistic Regression, Random Forest, Gradient Boosting)...")
    print("[+] Running 5-Fold Stratified Cross-Validation...")

    results = trainer.train_and_evaluate()
    elapsed = time.time() - t0

    print("\n" + "=" * 72)
    print(f"[*] Training complete in {elapsed:.2f} seconds!")
    print(f"[*] Total Training Samples: {results['train_samples']}")
    print(f"[*] Total Test Samples:     {results['test_samples']}")
    print(f"[*] Best Performing Model:  {results['best_model_name']}")
    print("=" * 72)

    print("\n" + "-" * 72)
    print(f"{'Model Name':<22} | {'Accuracy':<8} | {'ROC-AUC':<8} | {'F1-Score':<8} | {'CV ROC-AUC'}")
    print("-" * 72)

    for name, metrics in results["models"].items():
        print(
            f"{name:<22} | "
            f"{metrics['accuracy'] * 100:>7.2f}% | "
            f"{metrics['roc_auc']:>8.4f} | "
            f"{metrics['f1_score']:>8.4f} | "
            f"{metrics['cv_roc_auc_mean']:.4f} +/- {metrics['cv_roc_auc_std']:.4f}"
        )
    print("-" * 72)

    best_metrics = results["models"][results["best_model_name"]]
    cm = best_metrics["confusion_matrix"]
    print("\nConfusion Matrix for Best Model:")
    print(f"  True Negatives (Retained correctly): {cm['true_negative']}")
    print(f"  False Positives (Predicted Attrited, actually Stayed): {cm['false_positive']}")
    print(f"  False Negatives (Predicted Stayed, actually Left): {cm['false_negative']}")
    print(f"  True Positives (Attrited caught): {cm['true_positive']}")

    print("\nTop 5 Global Feature Drivers:")
    for i, item in enumerate(best_metrics["feature_importances"][:5], 1):
        print(f"  {i}. {item['feature']:<30} : {item['importance']:.4f}")

    print("\n[OK] Artifacts successfully serialized to backend/models/")


if __name__ == "__main__":
    main()
