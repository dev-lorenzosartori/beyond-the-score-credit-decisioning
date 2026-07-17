from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.metrics import ks_statistic, prior_probability_shift
from src.policy import EconomicAssumptions, expected_value, population_weights
from src.train_evaluate import select_champion


class MetricsAndPolicyTests(unittest.TestCase):
    def test_prior_shift_maps_sample_prior_to_target_prior(self):
        adjusted = prior_probability_shift([0.30], 0.30, 0.15)
        self.assertAlmostEqual(float(adjusted[0]), 0.15)
        monotonic = prior_probability_shift([0.1, 0.2, 0.4], 0.30, 0.15)
        self.assertTrue(np.all(np.diff(monotonic) > 0))

    def test_expected_value_rewards_lower_risk(self):
        assumptions = EconomicAssumptions()
        values = expected_value(np.array([0.05, 0.40]), np.array([3000, 3000]), assumptions)
        self.assertGreater(values[0], 0)
        self.assertLess(values[1], 0)

    def test_population_weights_recover_target_rate(self):
        y = np.array([0] * 70 + [1] * 30)
        weights = population_weights(y)
        self.assertAlmostEqual(float(np.average(y, weights=weights)), 0.15)

    def test_ks_uses_larger_scores_as_more_bad_risk(self):
        self.assertAlmostEqual(ks_statistic([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]), 1.0)

    def test_champion_selection_uses_validation_value_only(self):
        policies = pd.DataFrame(
            [
                {"model": "A", "split": "validation", "policy": "Expected-value policy", "value_units_per_100_applications": 10},
                {"model": "B", "split": "validation", "policy": "Expected-value policy", "value_units_per_100_applications": 20},
                {"model": "A", "split": "test", "policy": "Expected-value policy", "value_units_per_100_applications": 999},
            ]
        )
        self.assertEqual(select_champion(policies), "B")


if __name__ == "__main__":
    unittest.main()
