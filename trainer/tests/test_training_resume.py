from __future__ import annotations

import unittest

from rttrainer.training.runner import (
    correlation_coefficient,
    default_learning_rate_plateau_patience,
    numeric_metrics,
    recurrent_context_training_enabled,
    recurrent_context_training_multiplier,
    resolve_learning_rate_schedule,
    resolve_resume_learning_rate,
    resolve_training_loss_name,
    state_reset_diagnostic,
    target_epoch_count,
    validation_selection_metrics,
)
from rttrainer.models.presets import get_preset


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
        self.assertEqual(schedule["monitor"], "validation_score")

    def test_resume_learning_rate_does_not_increase_checkpoint_rate(self) -> None:
        learning_rate = resolve_resume_learning_rate(
            {},
            requested_learning_rate=0.001,
            resumed_checkpoint={"learning_rate": 0.000125},
        )

        self.assertEqual(learning_rate, 0.000125)

    def test_old_resume_checkpoint_uses_conservative_learning_rate(self) -> None:
        learning_rate = resolve_resume_learning_rate(
            {},
            requested_learning_rate=0.001,
            resumed_checkpoint={},
        )

        self.assertEqual(learning_rate, 0.0001)

    def test_training_loss_defaults_to_mse(self) -> None:
        self.assertEqual(resolve_training_loss_name({}), "mse")
        self.assertEqual(resolve_training_loss_name({"loss": "esr"}), "esr")
        self.assertEqual(resolve_training_loss_name({"loss": "hf_mse"}), "preemphasis_mse")
        self.assertEqual(
            resolve_training_loss_name({"loss": "mrstft_mse"}),
            "mrstft_preemphasis",
        )
        self.assertEqual(resolve_training_loss_name({"loss": "mean_squared_error"}), "mse")
        self.assertEqual(
            resolve_training_loss_name({}, get_preset("conv1d_stack_prelu")),
            "preemphasis_mse",
        )
        self.assertEqual(
            resolve_training_loss_name({}, get_preset("wavenet_tcn")),
            "mrstft_preemphasis",
        )
        self.assertEqual(
            resolve_training_loss_name({}, get_preset("wavenet_tcn_fast")),
            "mrstft_preemphasis",
        )
        self.assertEqual(
            resolve_training_loss_name({}, get_preset("wavenet_tcn_balanced")),
            "mrstft_preemphasis",
        )
        self.assertEqual(
            resolve_training_loss_name({}, get_preset("wavenet_tcn_quality")),
            "mrstft_preemphasis",
        )

    def test_training_loss_rejects_unknown_values(self) -> None:
        with self.assertRaises(ValueError):
            resolve_training_loss_name({"loss": "magic"})

    def test_recurrent_context_training_defaults_to_recurrent_presets(self) -> None:
        self.assertTrue(
            recurrent_context_training_enabled({}, get_preset("conv_gru_hybrid"))
        )
        self.assertFalse(
            recurrent_context_training_enabled({}, get_preset("conv1d_light"))
        )
        self.assertFalse(
            recurrent_context_training_enabled(
                {"recurrent_context_training_enabled": False},
                get_preset("lstm_standard"),
            )
        )

    def test_recurrent_context_multiplier_is_bounded(self) -> None:
        self.assertEqual(recurrent_context_training_multiplier({}), 4)
        self.assertEqual(
            recurrent_context_training_multiplier({"recurrent_context_multiplier": 0}),
            1,
        )
        self.assertEqual(
            recurrent_context_training_multiplier({"recurrent_context_multiplier": 200}),
            16,
        )

    def test_validation_score_penalizes_underpowered_stream_prediction(self) -> None:
        muted = validation_selection_metrics(
            {"esr": 1.0, "mae": 0.1, "rmse": 0.1},
            {"esr": 0.9, "mae": 0.1, "rmse": 0.1},
            stream_prediction=[0.0, 0.0, 0.0, 0.0],
            stream_target=[0.25, -0.25, 0.25, -0.25],
        )
        audible = validation_selection_metrics(
            {"esr": 1.01, "mae": 0.1, "rmse": 0.1},
            {"esr": 0.7, "mae": 0.1, "rmse": 0.1},
            stream_prediction=[0.2, -0.2, 0.2, -0.2],
            stream_target=[0.25, -0.25, 0.25, -0.25],
        )

        self.assertGreater(muted["underpowered_prediction_penalty"], 0.0)
        self.assertLess(audible["validation_score"], muted["validation_score"])

    def test_correlation_coefficient_detects_inverted_and_matched_signals(self) -> None:
        target = [0.0, 0.25, -0.25, 0.5, -0.5]

        self.assertAlmostEqual(correlation_coefficient(target, target), 1.0)
        self.assertAlmostEqual(
            correlation_coefficient(target, [-sample for sample in target]),
            -1.0,
        )

    def test_state_reset_diagnostic_flags_recurrent_drift(self) -> None:
        target = [0.0, 0.4, -0.4, 0.3, -0.3, 0.2, -0.2, 0.1]
        continuous = [0.0 for _sample in target]
        chunk_reset = [sample * 0.8 for sample in target]

        diagnostic = state_reset_diagnostic(
            get_preset("conv_gru_hybrid"),
            target=target,
            continuous_prediction=continuous,
            chunk_reset_prediction=chunk_reset,
            chunk_size=4,
            sample_rate=48_000,
        )

        self.assertTrue(diagnostic["applies"])
        self.assertEqual(diagnostic["verdict"], "state_drift_suspected")
        self.assertGreater(float(diagnostic["esr_delta"]), 0.1)
        self.assertGreater(float(diagnostic["chunk_reset_correlation"]), 0.25)

    def test_state_reset_diagnostic_marks_conv1d_as_finite_memory(self) -> None:
        target = [0.0, 0.4, -0.4, 0.3]

        diagnostic = state_reset_diagnostic(
            get_preset("conv1d_bn_prelu"),
            target=target,
            continuous_prediction=target,
            chunk_reset_prediction=target,
            chunk_size=4,
            sample_rate=48_000,
        )

        self.assertFalse(diagnostic["applies"])
        self.assertEqual(diagnostic["verdict"], "finite_memory")

    def test_state_reset_diagnostic_marks_wavenet_presets_as_finite_memory(self) -> None:
        target = [0.0, 0.4, -0.4, 0.3]

        for preset_id in (
            "wavenet_tcn_fast",
            "wavenet_tcn",
            "wavenet_tcn_balanced",
            "wavenet_tcn_quality",
        ):
            with self.subTest(preset=preset_id):
                diagnostic = state_reset_diagnostic(
                    get_preset(preset_id),
                    target=target,
                    continuous_prediction=target,
                    chunk_reset_prediction=target,
                    chunk_size=4,
                    sample_rate=48_000,
                )

                self.assertFalse(diagnostic["applies"])
                self.assertEqual(diagnostic["verdict"], "finite_memory")


if __name__ == "__main__":
    unittest.main()
