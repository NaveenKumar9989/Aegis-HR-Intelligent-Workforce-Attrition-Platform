"""
Unit Tests for DataPreprocessor.
"""

import unittest
import numpy as np
from backend.pipeline.data_loader import DataLoader
from backend.pipeline.preprocessor import DataPreprocessor
from backend.config import DATA_PATH


class TestDataPreprocessor(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        loader = DataLoader(DATA_PATH)
        raw_df = loader.load_raw_data()
        cls.X, cls.y = loader.clean_and_prepare(raw_df)

    def setUp(self):
        self.preprocessor = DataPreprocessor()

    def test_fit_transform(self):
        X_trans = self.preprocessor.fit_transform(self.X)
        self.assertIsInstance(X_trans, np.ndarray)
        self.assertEqual(X_trans.shape[0], len(self.X))
        self.assertTrue(self.preprocessor.is_fitted)
        self.assertGreater(len(self.preprocessor.feature_names_), len(self.preprocessor.numerical_cols))

    def test_transform_unfitted_raises_error(self):
        with self.assertRaises(RuntimeError):
            self.preprocessor.transform(self.X)

    def test_save_and_load_preprocessor(self):
        import tempfile
        from pathlib import Path

        self.preprocessor.fit(self.X)
        expected_feature_count = len(self.preprocessor.feature_names_)

        with tempfile.TemporaryDirectory() as tmpdir:
            temp_path = Path(tmpdir) / "test_prep.joblib"
            self.preprocessor.save(temp_path)
            self.assertTrue(temp_path.exists())

            loaded = DataPreprocessor.load(temp_path)
            self.assertTrue(loaded.is_fitted)
            self.assertEqual(len(loaded.feature_names_), expected_feature_count)

            # Check that transformed arrays match
            trans_orig = self.preprocessor.transform(self.X.iloc[:5])
            trans_loaded = loaded.transform(self.X.iloc[:5])
            np.testing.assert_allclose(trans_orig, trans_loaded)


if __name__ == "__main__":
    unittest.main()
