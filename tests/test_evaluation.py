import unittest

import numpy as np
import pandas as pd

from src.evaluation import baseline_costs, chronological_split, operating_cost, recall_at_k


class EvaluationTests(unittest.TestCase):
    def test_operating_cost_charges_every_alert_and_actual_missed_amount(self):
        result = operating_cost(
            y_true=[0, 1, 0, 1],
            scores=[0.1, 0.9, 0.8, 0.2],
            amounts=[10, 100, 20, 50],
            threshold=0.5,
            review_cost_per_alert=5,
        )
        self.assertEqual(result["alerts"], 2)
        self.assertEqual(result["true_positives"], 1)
        self.assertEqual(result["false_positives"], 1)
        self.assertEqual(result["false_negatives"], 1)
        self.assertEqual(result["review_cost_eur"], 10)
        self.assertEqual(result["missed_fraud_loss_eur"], 50)
        self.assertEqual(result["total_cost_eur"], 60)

    def test_baselines_are_not_the_same_comparison(self):
        result = baseline_costs(
            y_true=[0, 1, 0, 1],
            amounts=[10, 100, 20, 50],
            review_cost_per_alert=5,
        )
        self.assertEqual(result["review_nothing_eur"], 150)
        self.assertEqual(result["review_everything_eur"], 20)

    def test_recall_at_k(self):
        self.assertAlmostEqual(
            recall_at_k([0, 1, 0, 1], [0.1, 0.9, 0.8, 0.2], k=2),
            0.5,
        )

    def test_chronological_split_orders_time(self):
        df = pd.DataFrame({"Time": [3, 1, 4, 2, 5], "Class": [0, 1, 0, 0, 1]})
        train, validation, test = chronological_split(
            df,
            train_fraction=0.60,
            validation_fraction=0.20,
        )
        np.testing.assert_array_equal(train["Time"].to_numpy(), [1, 2, 3])
        np.testing.assert_array_equal(validation["Time"].to_numpy(), [4])
        np.testing.assert_array_equal(test["Time"].to_numpy(), [5])

    def test_chronological_split_keeps_equal_timestamps_together(self):
        df = pd.DataFrame(
            {
                "Time": [1, 2, 3, 4, 4, 5, 6],
                "Class": [0, 0, 1, 0, 1, 0, 1],
            }
        )
        train, validation, test = chronological_split(
            df,
            train_fraction=0.50,
            validation_fraction=0.25,
        )
        self.assertLess(train["Time"].max(), validation["Time"].min())
        self.assertLess(validation["Time"].max(), test["Time"].min())


if __name__ == "__main__":
    unittest.main()
