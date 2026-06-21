from __future__ import annotations

import unittest

from rttrainer.training.runner import (
    default_learning_rate_plateau_patience,
    numeric_metrics,
    resolve_learning_rate_schedule,
    target_epoch_count,
)


class TrainingResumeTests(unittest.TestCase):
    def test_resume_from_previous_run_adds_epochs_to_checkpoint_epoch(self) -> None:
        target_epochs = target_epoch_count(
            {"resume_epochs_are_additional": True},
            resumed_epoch=98,
            requested_epochs=40,
        )

        self.assertEqual(target_epochs, 138)

    def test_interrupted_run_resume_keeps_total_epoch_target(self) -> None:
        target_epochs = target_epoch_count({}, resumed_epoch=18, requested_epochs=40)

        self.assertEqual(target_epochs, 40)

    def test_numeric_metrics_discards_non_numeric_metadata(self) -> None:
        metrics = numeric_metrics({"esr": 0.25, "note": "best", "epoch": 12})

        self.assertEqual(metrics, {"esr": 0.25, "epoch": 12.0})

    def test_learning_rate_plateau_patience_precedes_early_stopping(self) -> None:
        self.assertEqual(default_learning_rate_plateau_patience(0), 5)
        self.assertEqual(default_learning_rate_plateau_patience(1), 1)
        self.assertEqual(default_learning_rate_plateau_patience(6), 3)
        self.assertEqual(default_learning_rate_plateau_patience(40), 10)

    def test_learning_rate_schedule_normalizes_values(self) -> None:
        schedule = resolve_learning_rate_schedule(
            {
                "learning_rate_plateau_factor": 2.0,
                "learning_rate_plateau_patience": 0,
                "learning_rate_plateau_min_delta": float("nan"),
                "min_learning_rate": 0.1,
            },
            initial_learning_rate=0.001,
            early_stopping_patience=8,
            early_stopping_min_delta=0.0001,
        )

        self.assertFalse(schedule["enabled"])
        self.assertEqual(schedule["factor"], 0.99)
        self.assertEqual(schedule["patience"], 1)
        self.assertEqual(schedule["min_delta"], 0.0001)
        self.assertEqual(schedule["min_learning_rate"], 0.001)


if __name__ == "__main__":
    unittest.main()
