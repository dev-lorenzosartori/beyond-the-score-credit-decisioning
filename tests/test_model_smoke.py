from __future__ import annotations

import unittest

import numpy as np

from src.data import load_credit_data, model_xy, split_credit_data
from src.train_evaluate import build_models


class ModelSmokeTests(unittest.TestCase):
    def test_models_produce_finite_probabilities(self):
        splits = split_credit_data(load_credit_data())
        X_train, y_train = model_xy(splits.train)
        X_validation, _ = model_xy(splits.validation)
        for model in build_models().values():
            model.fit(X_train, y_train)
            probabilities = model.predict_proba(X_validation.iloc[:12])[:, 1]
            self.assertEqual(probabilities.shape, (12,))
            self.assertTrue(np.isfinite(probabilities).all())
            self.assertTrue(((probabilities >= 0) & (probabilities <= 1)).all())


if __name__ == "__main__":
    unittest.main()
