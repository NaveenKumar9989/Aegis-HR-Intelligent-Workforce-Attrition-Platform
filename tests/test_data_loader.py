"""
Unit Tests for DataLoader & Data Validator.
"""

import unittest
import pandas as pd
from backend.pipeline.data_loader import DataLoader
from backend.config import DATA_PATH, TARGET_COL, DROP_COLS


class TestDataLoader(unittest.TestCase):

    def setUp(self):
        self.loader = DataLoader(DATA_PATH)

    def test_load_raw_data_success(self):
        df = self.loader.load_raw_data()
        self.assertIsInstance(df, pd.DataFrame)
        self.assertGreater(len(df), 1000)
        self.assertIn(TARGET_COL, df.columns)

    def test_validate_schema_valid(self):
        df = self.loader.load_raw_data()
        is_valid = self.loader.validate_schema(df, require_target=True)
        self.assertTrue(is_valid)

    def test_validate_schema_missing_column(self):
        df = self.loader.load_raw_data()
        df_incomplete = df.drop(columns=["Age"])
        with self.assertRaises(ValueError) as ctx:
            self.loader.validate_schema(df_incomplete, require_target=True)
        self.assertIn("Missing required feature columns", str(ctx.exception))

    def test_clean_and_prepare(self):
        raw_df = self.loader.load_raw_data()
        X, y = self.loader.clean_and_prepare(raw_df, require_target=True)

        self.assertIsNotNone(y)
        self.assertEqual(len(X), len(y))
        # Ensure invariant drop columns are removed
        for col in DROP_COLS:
            self.assertNotIn(col, X.columns)
        self.assertNotIn(TARGET_COL, X.columns)
        # Ensure binary target
        self.assertTrue(set(y.unique()).issubset({0, 1}))

    def test_get_dataset_stats(self):
        stats = self.loader.get_dataset_stats()
        self.assertIn("total_employees", stats)
        self.assertIn("attrition_rate", stats)
        self.assertIn("departments", stats)
        self.assertIn("overtime", stats)
        self.assertGreater(stats["total_employees"], 0)
        self.assertGreater(len(stats["departments"]), 0)


if __name__ == "__main__":
    unittest.main()
