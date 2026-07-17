from __future__ import annotations

import unittest

from src.config import AUDIT_ONLY_COLUMNS, MODEL_FEATURES
from src.data import file_sha256, load_credit_data, split_credit_data


class CreditDataContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frame = load_credit_data()
        cls.splits = split_credit_data(cls.frame)

    def test_source_hash_and_grain(self):
        self.assertEqual(
            file_sha256(),
            "baa78cca9b7c46631b9d941ed358595b5334e35270b922950e742130617c55f3",
        )
        self.assertEqual(self.frame.shape, (1000, 22))
        self.assertFalse(self.frame.application_id.duplicated().any())
        self.assertFalse(self.frame.isna().any().any())

    def test_split_contract(self):
        expected = {"train": (600, 180), "validation": (200, 60), "test": (200, 60)}
        for name, split in (
            ("train", self.splits.train),
            ("validation", self.splits.validation),
            ("test", self.splits.test),
        ):
            self.assertEqual((len(split), int(split.bad_credit.sum())), expected[name])
        all_ids = set(self.splits.train.application_id)
        self.assertFalse(all_ids & set(self.splits.validation.application_id))
        self.assertFalse(all_ids & set(self.splits.test.application_id))

    def test_audit_columns_never_enter_model(self):
        self.assertTrue(set(AUDIT_ONLY_COLUMNS).isdisjoint(MODEL_FEATURES))
        self.assertEqual(len(MODEL_FEATURES), 17)


if __name__ == "__main__":
    unittest.main()
