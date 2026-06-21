from __future__ import annotations

import unittest

from rttrainer.training.runner import numeric_metrics, target_epoch_count


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


if __name__ == "__main__":
    unittest.main()
