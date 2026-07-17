from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.fairness import fairness_audit, wilson_interval
from src.monitoring import categorical_psi, drift_severity, numeric_psi


class MonitoringAndFairnessTests(unittest.TestCase):
    def test_psi_is_zero_for_identical_population(self):
        numeric = pd.Series(np.arange(100))
        category = pd.Series(["a", "b"] * 50)
        self.assertAlmostEqual(numeric_psi(numeric, numeric), 0.0)
        self.assertAlmostEqual(categorical_psi(category, category), 0.0)

    def test_drift_thresholds(self):
        self.assertEqual(drift_severity(0.05), "stable")
        self.assertEqual(drift_severity(0.15), "watch")
        self.assertEqual(drift_severity(0.30), "action")

    def test_wilson_interval_contains_observed_rate(self):
        low, high = wilson_interval(5, 10)
        self.assertLess(low, 0.5)
        self.assertGreater(high, 0.5)

    def test_small_groups_are_flagged(self):
        frame = pd.DataFrame(
            {
                "age": [20, 20, 40, 40],
                "foreign_worker": [1, 2, 2, 2],
                "bad_credit": [0, 1, 0, 1],
            }
        )
        result = fairness_audit(frame, np.array([1, 0, 1, 0]), np.array([0.1, 0.8, 0.2, 0.7]))
        self.assertTrue(result.small_sample_warning.all())


if __name__ == "__main__":
    unittest.main()
