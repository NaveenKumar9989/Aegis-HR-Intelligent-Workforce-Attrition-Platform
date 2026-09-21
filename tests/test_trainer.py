"""
Unit Tests for ModelTrainer and Evaluator.
"""

import unittest
from backend.pipeline.trainer import ModelTrainer
from backend.config import BEST_MODEL_PATH, PREPROCESSOR_PATH, METRICS_PATH


class TestModelTrainer(unittest.TestCase):

    def setUp(self):
        self.trainer = ModelTrainer()

    def test_get_candidate_models(self):
        models = self.trainer.get_candidate_models()
        self.assertIn("Logistic Regression", models)
        self.assertIn("Random Forest", models)
        self.assertIn("Gradient Boosting", models)

    def test_artifacts_exist_after_training(self):
        self.assertTrue(BEST_MODEL_PATH.exists(), "Best model artifact missing")
        self.assertTrue(PREPROCESSOR_PATH.exists(), "Preprocessor artifact missing")
        self.assertTrue(METRICS_PATH.exists(), "Metrics JSON missing")


if __name__ == "__main__":
    unittest.main()
